// C++ port of qlio/ekf.py: StochasticCloningEKF (stochastic-cloning EKF with
// learned displacement updates). Semantics and order of operations match the
// Python/numpy implementation (float64 throughout).
//
// Error state: [dtheta(3), dv(3), dp(3), dbg(3), dba(3)] + 6 per clone
// [dtheta_c, dp_c]. Attitude error is right-multiplicative: R = R_hat exp(dtheta).

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <cmath>
#include <deque>
#include <optional>
#include <stdexcept>
#include <utility>

namespace py = pybind11;

using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;
using MatX = Eigen::MatrixXd;
using VecX = Eigen::VectorXd;
using Mat15 = Eigen::Matrix<double, 15, 15>;

static const Vec3 GRAVITY(0.0, 0.0, -9.81);
static const Vec3 E3(0.0, 0.0, 1.0);

// ---- geometry helpers (mirroring qlio/geometry.py) ----------------------

static Mat3 skew(const Vec3& v) {
    Mat3 S;
    S << 0.0, -v(2), v(1),
         v(2), 0.0, -v(0),
        -v(1), v(0), 0.0;
    return S;
}

static Mat3 so3_exp(const Vec3& w) {
    const double theta = w.norm();
    const Mat3 W = skew(w);
    const Mat3 WW = W * W;
    if (theta < 1e-8) {
        return Mat3(Mat3::Identity() + W + 0.5 * WW);
    }
    const double s = std::sin(theta);
    const double c = std::cos(theta);
    return Mat3(Mat3::Identity() + (s / theta) * W
                + ((1.0 - c) / std::pow(theta, 2.0)) * WW);
}

static Mat3 so3_right_jacobian_inv(const Vec3& w) {
    const double theta = w.norm();
    const Mat3 W = skew(w);
    if (theta < 1e-8) {
        return Mat3(Mat3::Identity() + 0.5 * W);
    }
    const double a = 1.0 / std::pow(theta, 2.0)
                     - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
    const Mat3 WW = W * W;
    return Mat3(Mat3::Identity() + 0.5 * W + a * WW);
}

static double yaw_of(const Mat3& R) {
    return std::atan2(R(1, 0), R(0, 0));
}

// Yaw-only rotation extracted from R (gravity-aligned body frame).
static Mat3 gravity_aligned_frame(const Mat3& R) {
    const double y = yaw_of(R);
    const double c = std::cos(y), s = std::sin(y);
    Mat3 G;
    G << c, -s, 0.0,
         s,  c, 0.0,
         0.0, 0.0, 1.0;
    return G;
}

// Row vector c such that d(yaw) = c^T phi for a world-frame perturbation exp(phi)R.
static Vec3 yaw_world_jacobian(const Mat3& R) {
    const double d = R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0);
    if (d < 1e-12) return Vec3(0.0, 0.0, 1.0);
    return Vec3(-R(2, 0) * R(0, 0) / d, -R(2, 0) * R(1, 0) / d, 1.0);
}

// ---- config / clone ------------------------------------------------------

struct EKFConfig {
    double sigma_g = 2e-3;
    double sigma_a = 2e-2;
    double sigma_bg = 1e-4;
    double sigma_ba = 1e-3;
    double init_sigma_theta = 1e-2;
    double init_sigma_v = 1e-1;
    double init_sigma_p = 1e-3;
    double init_sigma_bg = 5e-3;
    double init_sigma_ba = 3e-2;
    int max_clones = 2;
    std::optional<double> chi2_gate = 30.0;   // nullopt disables the gate
    double cov_inflation = 1.0;
    bool use_fej = true;
};

struct Clone {
    Mat3 R;
    Vec3 p;
    double t;
    long cid;
    Mat3 R_g;   // frozen at cloning under FEJ
};

// ---- filter ---------------------------------------------------------------

class StochasticCloningEKF {
public:
    EKFConfig cfg;
    Mat3 R = Mat3::Identity();
    Vec3 v = Vec3::Zero();
    Vec3 p = Vec3::Zero();
    Vec3 bg = Vec3::Zero();
    Vec3 ba = Vec3::Zero();
    MatX P;
    std::deque<Clone> clones;
    long next_cid = 0;

    explicit StochasticCloningEKF(const EKFConfig& c) : cfg(c) {
        P = MatX::Zero(15, 15);
        for (int i = 0; i < 3; ++i) {
            P(i, i) = c.init_sigma_theta * c.init_sigma_theta;
            P(3 + i, 3 + i) = c.init_sigma_v * c.init_sigma_v;
            P(6 + i, 6 + i) = c.init_sigma_p * c.init_sigma_p;
            P(9 + i, 9 + i) = c.init_sigma_bg * c.init_sigma_bg;
            P(12 + i, 12 + i) = c.init_sigma_ba * c.init_sigma_ba;
        }
    }

    int n_clones() const { return static_cast<int>(clones.size()); }
    int dim() const { return 15 + 6 * n_clones(); }

    void set_state(const std::optional<Mat3>& R_, const std::optional<Vec3>& v_,
                   const std::optional<Vec3>& p_, const std::optional<Vec3>& bg_,
                   const std::optional<Vec3>& ba_) {
        if (R_) R = *R_;
        if (v_) v = *v_;
        if (p_) p = *p_;
        if (bg_) bg = *bg_;
        if (ba_) ba = *ba_;
    }

    static void symmetrize(MatX& M) {
        MatX Mt = M.transpose();
        M = 0.5 * (M + Mt);
    }

    // ---- propagation ------------------------------------------------------
    void propagate(const Vec3& gyr_b, const Vec3& acc_b, double dt) {
        const EKFConfig& c = cfg;
        const double dt2 = std::pow(dt, 2.0);
        const double dt3 = std::pow(dt, 3.0);
        const Vec3 w = gyr_b - bg;
        const Vec3 a = acc_b - ba;
        const Vec3 Ra = R * a;

        const Mat3 dR = so3_exp(w * dt);
        p = p + v * dt + (0.5 * (Ra + GRAVITY)) * dt2;
        v = v + (Ra + GRAVITY) * dt;
        R = R * dR;   // F below is built from the post-update R

        Mat15 F = Mat15::Identity();
        const Mat3 Sa = skew(a);
        const Mat3 RSa = R * Sa;
        F.block<3, 3>(0, 0) = dR.transpose();
        F.block<3, 3>(0, 9) = -so3_right_jacobian_inv(w * dt).transpose() * dt;
        F.block<3, 3>(3, 0) = -RSa * dt;
        F.block<3, 3>(3, 12) = -R * dt;
        F.block<3, 3>(6, 3) = Mat3::Identity() * dt;
        F.block<3, 3>(6, 0) = (-0.5 * RSa) * dt2;
        F.block<3, 3>(6, 12) = (-0.5 * R) * dt2;

        Mat15 Qd = Mat15::Zero();
        for (int i = 0; i < 3; ++i) {
            Qd(i, i) = c.sigma_g * c.sigma_g * dt;
            Qd(3 + i, 3 + i) = c.sigma_a * c.sigma_a * dt;
            Qd(6 + i, 6 + i) = c.sigma_a * c.sigma_a * dt3 / 3.0;
            Qd(9 + i, 9 + i) = c.sigma_bg * c.sigma_bg * dt;
            Qd(12 + i, 12 + i) = c.sigma_ba * c.sigma_ba * dt;
        }

        // Clones are static under propagation: only the core block and the
        // core/clone cross-terms change.
        const int nc = dim() - 15;
        const Mat15 Pxx = P.topLeftCorner<15, 15>();
        P.topLeftCorner<15, 15>() = F * Pxx * F.transpose() + Qd;
        if (nc > 0) {
            MatX Pxc = F * P.topRightCorner(15, nc);
            P.topRightCorner(15, nc) = Pxc;
            P.bottomLeftCorner(nc, 15) = Pxc.transpose();
        }
        symmetrize(P);
    }

    // ---- cloning ------------------------------------------------------
    long clone(double t) {
        if (n_clones() >= cfg.max_clones) marginalize_oldest();
        const int n = dim();
        MatX Pn(n + 6, n + 6);
        Pn.topLeftCorner(n, n) = P;
        Pn.block(0, n, n, 3) = P.middleCols(0, 3);       // P J^T
        Pn.block(0, n + 3, n, 3) = P.middleCols(6, 3);
        Pn.block(n, 0, 3, n) = P.middleRows(0, 3);       // J P
        Pn.block(n + 3, 0, 3, n) = P.middleRows(6, 3);
        Pn.block(n, n, 3, 3) = P.block(0, 0, 3, 3);      // J P J^T
        Pn.block(n, n + 3, 3, 3) = P.block(0, 6, 3, 3);
        Pn.block(n + 3, n, 3, 3) = P.block(6, 0, 3, 3);
        Pn.block(n + 3, n + 3, 3, 3) = P.block(6, 6, 3, 3);
        P = Pn;
        symmetrize(P);
        const long cid = next_cid++;
        clones.push_back(Clone{R, p, t, cid, gravity_aligned_frame(R)});
        return cid;
    }

    void marginalize_oldest() {
        if (clones.empty()) return;
        const int n = dim();
        const int rest = n - 21;
        MatX Pn(n - 6, n - 6);
        Pn.topLeftCorner(15, 15) = P.topLeftCorner(15, 15);
        if (rest > 0) {
            Pn.block(0, 15, 15, rest) = P.block(0, 21, 15, rest);
            Pn.block(15, 0, rest, 15) = P.block(21, 0, rest, 15);
            Pn.block(15, 15, rest, rest) = P.block(21, 21, rest, rest);
        }
        P = Pn;
        clones.pop_front();
    }

    std::optional<int> clone_index(long cid) const {
        for (int k = 0; k < n_clones(); ++k) {
            if (clones[k].cid == cid) return k;
        }
        return std::nullopt;
    }

    std::optional<long> oldest_cid() const {
        if (clones.empty()) return std::nullopt;
        return clones.front().cid;
    }

    // ---- measurement update ---------------------------------------
    std::pair<MatX, Vec3> displacement_jacobian(int k) const {
        const Clone& cl = clones.at(static_cast<size_t>(k));
        const Mat3& R_g = cl.R_g;
        MatX H = MatX::Zero(3, dim());
        H.block<3, 3>(0, 6) = R_g.transpose();
        const int s = 15 + 6 * k;
        H.block<3, 3>(0, s + 3) = -R_g.transpose();
        if (!cfg.use_fej) {
            // Legacy frame handling: model depends on clone yaw.
            const Vec3 dp = p - cl.p;
            const Eigen::RowVector3d dyaw_dtheta =
                yaw_world_jacobian(cl.R).transpose() * cl.R;
            const Mat3 M = R_g.transpose() * skew(E3);
            const Vec3 u = M * dp;
            H.block<3, 3>(0, s) = -(u * dyaw_dtheta);
        }
        const Vec3 h = R_g.transpose() * (p - cl.p);
        return {H, h};
    }

    py::dict update_displacement(const Vec3& z, const py::array_t<double>& cov, int k) {
        auto Hh = displacement_jacobian(k);
        const MatX& H = Hh.first;
        const Vec3& h = Hh.second;
        const Vec3 r = z - h;

        Mat3 Sigma;
        py::array_t<double, py::array::c_style | py::array::forcecast> covc(cov);
        if (covc.ndim() == 1) {
            if (covc.shape(0) != 3) throw std::invalid_argument("cov must have 3 entries");
            Sigma = Mat3::Zero();
            auto b = covc.unchecked<1>();
            for (int i = 0; i < 3; ++i) Sigma(i, i) = b(i) * cfg.cov_inflation;
        } else if (covc.ndim() == 2) {
            if (covc.shape(0) != 3 || covc.shape(1) != 3)
                throw std::invalid_argument("cov must be 3x3");
            auto b = covc.unchecked<2>();
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) Sigma(i, j) = b(i, j) * cfg.cov_inflation;
        } else {
            throw std::invalid_argument("cov must be 1-d or 2-d");
        }

        const MatX HP = H * P;
        const Mat3 S = HP * H.transpose() + Sigma;
        const double nis = r.dot(S.partialPivLu().solve(r));
        py::dict out;
        out["nis"] = nis;
        out["residual"] = r;
        if (cfg.chi2_gate.has_value() && nis > *cfg.chi2_gate) {
            out["accepted"] = false;
            return out;
        }

        const MatX K = Mat3(S.transpose()).partialPivLu().solve(HP).transpose();
        const VecX dx = K * r;
        // Joseph form expanded: P - A - A^T + K S K^T, then symmetrize.
        const MatX A = K * HP;
        const MatX At = A.transpose();
        const MatX KSKt = K * S * K.transpose();
        P = P - A - At + KSKt;
        symmetrize(P);
        inject(dx);
        out["accepted"] = true;
        return out;
    }

    py::dict update_batch(const MatX& H, const VecX& r, const MatX& Rm) {
        const MatX HP = H * P;
        const MatX S = HP * H.transpose() + Rm;
        const MatX K = MatX(S.transpose()).partialPivLu().solve(HP).transpose();
        const VecX dx = K * r;
        const MatX A = K * HP;
        const MatX At = A.transpose();
        const MatX KSKt = K * S * K.transpose();
        P = P - A - At + KSKt;
        symmetrize(P);
        inject(dx);
        py::dict out;
        out["rows"] = static_cast<int>(r.size());
        out["dx_norm"] = dx.norm();
        return out;
    }

    void inject(const VecX& dx) {
        R = R * so3_exp(dx.segment<3>(0));
        v = v + dx.segment<3>(3);
        p = p + dx.segment<3>(6);
        bg = bg + dx.segment<3>(9);
        ba = ba + dx.segment<3>(12);
        for (int k = 0; k < n_clones(); ++k) {
            Clone& cl = clones[k];
            const int s = 15 + 6 * k;
            cl.R = cl.R * so3_exp(dx.segment<3>(s));
            cl.p = cl.p + dx.segment<3>(s + 3);
            if (!cfg.use_fej) {
                // Legacy: re-linearise the frame; under FEJ it stays frozen.
                cl.R_g = gravity_aligned_frame(cl.R);
            }
        }
    }

    // ---- diagnostics ------------------------------------------------
    MatX pose_covariance() const {
        MatX C(6, 6);
        C.block<3, 3>(0, 0) = P.block(0, 0, 3, 3);
        C.block<3, 3>(0, 3) = P.block(0, 6, 3, 3);
        C.block<3, 3>(3, 0) = P.block(6, 0, 3, 3);
        C.block<3, 3>(3, 3) = P.block(6, 6, 3, 3);
        return C;
    }

    py::dict get_clone(int k) const {
        const Clone& cl = clones.at(static_cast<size_t>(k));
        py::dict d;
        d["R"] = cl.R;
        d["p"] = cl.p;
        d["t"] = cl.t;
        d["cid"] = cl.cid;
        d["R_g"] = cl.R_g;
        return d;
    }
};

// ---- bindings --------------------------------------------------------------

PYBIND11_MODULE(qlio_ekf, m) {
    m.doc() = "C++ (Eigen) port of qlio.ekf.StochasticCloningEKF";

    py::class_<StochasticCloningEKF>(m, "StochasticCloningEKF")
        .def(py::init([](double sigma_g, double sigma_a, double sigma_bg, double sigma_ba,
                         double init_sigma_theta, double init_sigma_v, double init_sigma_p,
                         double init_sigma_bg, double init_sigma_ba, int max_clones,
                         std::optional<double> chi2_gate, double cov_inflation, bool use_fej) {
                 EKFConfig c;
                 c.sigma_g = sigma_g;
                 c.sigma_a = sigma_a;
                 c.sigma_bg = sigma_bg;
                 c.sigma_ba = sigma_ba;
                 c.init_sigma_theta = init_sigma_theta;
                 c.init_sigma_v = init_sigma_v;
                 c.init_sigma_p = init_sigma_p;
                 c.init_sigma_bg = init_sigma_bg;
                 c.init_sigma_ba = init_sigma_ba;
                 c.max_clones = max_clones;
                 c.chi2_gate = chi2_gate;
                 c.cov_inflation = cov_inflation;
                 c.use_fej = use_fej;
                 return new StochasticCloningEKF(c);
             }),
             py::arg("sigma_g") = 2e-3, py::arg("sigma_a") = 2e-2,
             py::arg("sigma_bg") = 1e-4, py::arg("sigma_ba") = 1e-3,
             py::arg("init_sigma_theta") = 1e-2, py::arg("init_sigma_v") = 1e-1,
             py::arg("init_sigma_p") = 1e-3, py::arg("init_sigma_bg") = 5e-3,
             py::arg("init_sigma_ba") = 3e-2, py::arg("max_clones") = 2,
             py::arg("chi2_gate") = std::optional<double>(30.0),
             py::arg("cov_inflation") = 1.0, py::arg("use_fej") = true)
        .def_property_readonly("R", [](const StochasticCloningEKF& e) { return e.R; })
        .def_property_readonly("v", [](const StochasticCloningEKF& e) { return e.v; })
        .def_property_readonly("p", [](const StochasticCloningEKF& e) { return e.p; })
        .def_property_readonly("bg", [](const StochasticCloningEKF& e) { return e.bg; })
        .def_property_readonly("ba", [](const StochasticCloningEKF& e) { return e.ba; })
        .def_property_readonly("P", [](const StochasticCloningEKF& e) { return e.P; })
        .def_property_readonly("n_clones", &StochasticCloningEKF::n_clones)
        .def_property_readonly("dim", &StochasticCloningEKF::dim)
        .def_property_readonly("oldest_cid", &StochasticCloningEKF::oldest_cid)
        .def("set_state", &StochasticCloningEKF::set_state,
             py::arg("R") = py::none(), py::arg("v") = py::none(),
             py::arg("p") = py::none(), py::arg("bg") = py::none(),
             py::arg("ba") = py::none())
        .def("propagate", &StochasticCloningEKF::propagate,
             py::arg("gyr_b"), py::arg("acc_b"), py::arg("dt"))
        .def("clone", &StochasticCloningEKF::clone, py::arg("t"))
        .def("marginalize_oldest", &StochasticCloningEKF::marginalize_oldest)
        .def("clone_index", &StochasticCloningEKF::clone_index, py::arg("cid"))
        .def("displacement_jacobian", &StochasticCloningEKF::displacement_jacobian,
             py::arg("k"))
        .def("update_displacement", &StochasticCloningEKF::update_displacement,
             py::arg("z"), py::arg("cov"), py::arg("k") = 0)
        .def("update_batch", &StochasticCloningEKF::update_batch,
             py::arg("H"), py::arg("r"), py::arg("R"))
        .def("inject", &StochasticCloningEKF::inject, py::arg("dx"))
        .def("pose_covariance", &StochasticCloningEKF::pose_covariance)
        .def("get_clone", &StochasticCloningEKF::get_clone, py::arg("k"));
}
