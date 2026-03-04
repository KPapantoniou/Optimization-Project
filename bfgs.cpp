#include "utils.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include "bfgs.h"
#include "optimization.h"
#include "newtonTr.h"

using namespace std;

BFGS::BFGS(int maxIterations, double tolerance)
    : OptimizationAlgorithm(maxIterations, tolerance) {
}

double BFGS::objectiveFunction(const vector<double>& beta, const vector<array<double, 5>>& data) {
    evalCount++;
    double mse = 0.0;
    vector<double> denormalizedBeta = beta;
    denormalizeBeta(denormalizedBeta);
    vector<array<double, 5>> normilizedData = data;
    for (const auto& row : normilizedData) {
        double v = row[0];
        double theta = row[1];
        double T = row[2];
        double P = row[3];
        double E = row[4]; 
        double predicted = predictEnergy(denormalizedBeta, v, theta, T, P);
        mse += (-E + predicted) * (-E + predicted);
    }
    
    return mse / data.size();
}




double BFGS::lineSearch(const vector<double>& x, const vector<double>& p, const vector<array<double, 5>>& data) {
    const double c1 = 1e-4;
    const double c2 = 0.9;
    const double alpha_max_cap = 10.0;
    const int max_iters = 50;

    double phi0 = objectiveFunction(x, data);
    vector<double> grad0 = NewtonTr::gradient(x, data);
    double phi_prime0 = NewtonTr::dotProduct(grad0, p);
    if (phi_prime0 >= 0.0) {
        return 0.0;
    }

    double alpha_max = alpha_max_cap;
    for (size_t i = 0; i < x.size(); ++i) {
        if (p[i] > 0.0) {
            alpha_max = min(alpha_max, (1.0 - x[i]) / p[i]);
        } else if (p[i] < 0.0) {
            alpha_max = min(alpha_max, (0.0 - x[i]) / p[i]);
        }
    }
    if (alpha_max <= 0.0) {
        return 0.0;
    }

    double alpha_prev = 0.0;
    double alpha = min(1.0, alpha_max);
    double phi_prev = phi0;

    for (int i = 0; i < max_iters; ++i) {
        vector<double> x_new = NewtonTr::addVectors(x, scalarMultiplyVector(alpha, p));
        double phi_alpha = objectiveFunction(x_new, data);

        if ((phi_alpha > phi0 + c1 * alpha * phi_prime0) || (i > 0 && phi_alpha >= phi_prev)) {
            return zoom(alpha_prev, alpha, phi0, phi_prime0, c1, c2, data, x, p);
        }

        vector<double> grad = NewtonTr::gradient(x_new, data);
        double phi_prime_alpha = NewtonTr::dotProduct(grad, p);
        if (abs(phi_prime_alpha) <= -c2 * phi_prime0) {
            return alpha;
        }

        if (phi_prime_alpha >= 0.0) {
            return zoom(alpha, alpha_prev, phi0, phi_prime0, c1, c2, data, x, p);
        }

        alpha_prev = alpha;
        phi_prev = phi_alpha;
        alpha = min(2.0 * alpha, alpha_max);
    }

    return alpha;
}

double BFGS::interpolate(double alphaL, double alphaH){
    return 0.5*(alphaL+alphaH);
}

double BFGS:: zoom(double alphaL, double alphaH, double phi_zero, double phi_prime_zero, double c1, double c2, const vector<array<double, 5>>& data, const vector<double>& beta, const vector<double>& direction){
    while(true){
        double alphaJ = interpolate(alphaL,alphaH);
        vector<double> x_new = NewtonTr::addVectors(beta, scalarMultiplyVector(alphaJ, direction));
        double phiAlphaJ = objectiveFunction(x_new, data);
        vector<double> grad = NewtonTr::gradient(x_new, data);
        double phi_prime_alpha_j = NewtonTr::dotProduct(grad,direction);
        vector<double> x_alphaL = NewtonTr::addVectors(beta, scalarMultiplyVector(alphaL, direction));
        double phiAlphaL = objectiveFunction(x_alphaL, data);
        if((phiAlphaJ > phi_zero + c1*alphaJ*phi_prime_zero) || (phiAlphaJ >= phiAlphaL)){
            alphaH = alphaJ;
        }else{
           if(abs(phi_prime_alpha_j) <= -c2*phi_prime_zero){
               return alphaJ;
           }
              if(phi_prime_alpha_j*(alphaH-alphaL) >= 0){
                alphaH = alphaL;
              }
                alphaL = alphaJ;
        }
        if (abs(alphaH - alphaL) < 1e-8) {
            return alphaJ;
        }
    }
}


pair<vector<double>,double> BFGS::optimize(const vector<array<double, 5>>& data, const vector<double>& initialPoint) {
    evalCount = 0;
    vector<double> beta = initialPoint; 
    vector<double> grad(beta.size(), 0);
    grad = NewtonTr::gradient(beta, data);
    
    vector<vector<double>> hessian_matrix_inverse(beta.size(), vector<double>(beta.size(), 0.0));
    for (size_t i = 0; i < beta.size(); ++i) {
        hessian_matrix_inverse[i][i] = 1.0;
    }
    double k = 0;
    double alpha = 0;
    vector<double> sk(grad.size(), 0);
    vector<double> beta_old = beta;
    vector<double> yk(grad.size(), 0);
    

    while (NewtonTr::norm(grad) >= this->tolerance) {
            if (evalCount >= static_cast<size_t>(maxIterations)) {
                break;
            }
            beta_old = beta;

            vector<double> pk = NewtonTr::negateVector(NewtonTr::matrixVectorMultiplication(hessian_matrix_inverse,grad));

            alpha = lineSearch(beta,pk,data);
            if (alpha == 0.0) {
                for (size_t i = 0; i < hessian_matrix_inverse.size(); ++i) {
                    for (size_t j = 0; j < hessian_matrix_inverse.size(); ++j) {
                        hessian_matrix_inverse[i][j] = (i == j) ? 1.0 : 0.0;
                    }
                }
                pk = NewtonTr::negateVector(grad);
                alpha = lineSearch(beta, pk, data);
            }

            beta = NewtonTr::addVectors(beta,scalarMultiplyVector(alpha,pk));

            sk = NewtonTr::addVectors(beta,NewtonTr::negateVector(beta_old));

            yk = NewtonTr::addVectors(NewtonTr::gradient(beta,data),NewtonTr::negateVector(grad));

            if (NewtonTr::dotProduct(yk, sk) > 1e-12) {
                hessian_matrix_inverse = BFGSUpdate(hessian_matrix_inverse,sk,yk);
            } else {
                for (size_t i = 0; i < hessian_matrix_inverse.size(); ++i) {
                    for (size_t j = 0; j < hessian_matrix_inverse.size(); ++j) {
                        hessian_matrix_inverse[i][j] = (i == j) ? 1.0 : 0.0;
                    }
                }
            }

            grad = NewtonTr::gradient(beta, data);
            
            k++;
            
            if (NewtonTr::norm(grad) < this->tolerance) {
                break;
            }
            
        }
    return {beta, static_cast<double>(evalCount)};
}


vector<vector<double>> BFGS::BFGSUpdate(vector<vector<double>> B, vector<double> s, vector<double> y){
    double ys_dot = NewtonTr::dotProduct(y, s);
    if (abs(ys_dot) < 1e-10) { 
        return B; 
    }
    double rho = 1.0 / ys_dot;
    
    vector<vector<double>> I(B.size(), vector<double>(B.size(), 0.0));
    for (size_t i = 0; i < I.size(); ++i) {
        I[i][i] = 1.0;
    }

    vector<vector<double>> syT = outerProduct(s, y);
    vector<vector<double>> ysT = outerProduct(y, s);
    vector<vector<double>> ssT = outerProduct(s, s);

    vector<vector<double>> r_ysT = scalarMultiplyMatrix(ysT, rho);
    vector<vector<double>> r_syT = scalarMultiplyMatrix(syT, rho);
    vector<vector<double>> term_right = addMatrices(I,negateMatrix(r_ysT));
    vector<vector<double>> term_left = addMatrices(I,negateMatrix(r_syT));
    vector<vector<double>> term1 = matrixMultiplication(term_right, B);
    vector<vector<double>> B_new = matrixMultiplication(term1, term_left);

    return addMatrices(B_new, scalarMultiplyMatrix(ssT, rho));
}

vector<double>BFGS::scalarMultiplyVector(double scalar, const vector<double>& vec) {
    vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = scalar * vec[i];
    }
    return result;
}
vector<vector<double>>BFGS:: outerProduct(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size()) {
        throw invalid_argument("Vectors must have the same size for outer product.");
    }

    vector<vector<double>> result(a.size(), vector<double>(b.size()));

    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result[i][j] = a[i] * b[j];  
        }
    }

    return result;
}

vector<vector<double>>BFGS::negateMatrix(const vector<vector<double>>& matrix) {
    vector<vector<double>> result(matrix.size(), vector<double>(matrix[0].size()));

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[i][j] = -matrix[i][j];
        }
    }

    return result;
}

vector<vector<double>>BFGS::addMatrices(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        throw invalid_argument("Matrices must have the same size for addition.");
    }

    vector<vector<double>> result(A.size(), vector<double>(A[0].size()));

    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }

    return result;
}
vector<vector<double>>BFGS::subtractMatrices(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        throw invalid_argument("Matrices must have the same size for subtraction.");
    }

    vector<vector<double>> result(A.size(), vector<double>(A[0].size()));

    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }

    return result;
}
vector<vector<double>>BFGS::matrixMultiplication(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    if (A[0].size() != B.size()) {
        throw invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    vector<vector<double>> result(A.size(), vector<double>(B[0].size(), 0.0));

    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < B[0].size(); ++j) {
            for (size_t k = 0; k < A[0].size(); ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

vector<vector<double>> BFGS::scalarMultiplyMatrix(const vector<vector<double>>& matrix, double scalar) {
    vector<vector<double>> result(matrix.size(), vector<double>(matrix[0].size(), 0.0));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            result[i][j] = matrix[i][j] * scalar;
        }
    }
    return result;
}

