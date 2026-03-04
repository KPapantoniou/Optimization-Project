#include <vector>
#include <array>
#include "utils.h"
#include "newtonTr.h"
#include <iostream>
#include <cmath>
#include "optimization.h"
#include <limits>
#include "utils.h"
#include "bfgs.h"

using namespace std;
NewtonTr::NewtonTr(int maxIterations, double tolerance)
    : OptimizationAlgorithm(maxIterations, tolerance) {

}

double NewtonTr::objectiveFunction(const vector<double>& beta, const vector<array<double, 5>>& data) {
    evalCount++;
    double mse = 0.0;

    vector<double> denormalizedBeta = beta;
    vector<array<double, 5>> normilizedData = data;
    for (const auto& row : normilizedData) {
        double v = row[0];
        double theta = row[1];
        double T = row[2];
        double P = row[3];
        double E = row[4]; 
        double predicted = predictEnergy(denormalizedBeta, v, theta, T, P);
        mse += (predicted-E) * (predicted -E);
    }
    
    return mse / data.size();
    
}

vector<double> NewtonTr::gradient(const vector<double>& beta, const vector<array<double, 5>>& data) {
    size_t n = beta.size();
    double h = 1e-4; 
    vector<double> grad(n, 0.0);

    vector<array<double, 5>> normilizedData = data;
    vector<double> betaPerturbed = beta;
    vector<double> betaMinus = beta;
    double predictPlus = 0.0;
    double predictMinus = 0.0;

    for (const auto& row : normilizedData) {
        double predicted = predictEnergy(beta, row[0], row[1], row[2], row[3]);
        double residual = (predicted - row[4]);
        double v = row[0];
        double theta = row[1];
        double T = row[2];
        double P = row[3];
        double E = row[4]; 

        grad[0] += 2*residual*v*v;
        grad[1] += 2*residual*sin(theta);
        grad[2] += 2*residual*exp(beta[3]*T);
        grad[3] += 2*residual*beta[2]*T*exp(beta[3]*T);
        grad[4] += 2*residual*log(P);
    }
    for (size_t i = 0; i < n; ++i) {
        grad[i] /= normilizedData.size();
    }

    return grad;
}




vector<vector<double>> NewtonTr::hessian(const vector<double>& beta, const vector<array<double, 5>>& data) {
    
    size_t n = beta.size();
    double h = 1e-5; 
    vector<vector<double>> hessian(n, vector<double>(n, 0.0));
    vector<array<double, 5>> normilizedData = data;
    vector<double> energy_grad(n, 0.0);
    for(const auto& row : normilizedData){
        double v = row[0];
        double theta = row[1];
        double T = row[2];
        double P = row[3];
        double E = row[4];

        energy_grad[0] = v*v;
        energy_grad[1] = sin(theta);
        energy_grad[2] = exp(beta[3]*T);
        energy_grad[3] = beta[2]*T*exp(beta[3]*T);
        energy_grad[4] = log(P);

        double energy_diff = predictEnergy(beta, v, theta, T, P) - E;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                hessian[i][j] += energy_grad[i] * energy_grad[j];
            }
        }

        hessian[3][3] += energy_diff * beta[2] * T * T * exp(beta[3] * T);
        hessian[2][3] += energy_diff * T * exp(beta[3] * T);
        hessian[3][2] += energy_diff * T * exp(beta[3] * T);
    }
    for(size_t i = 0; i < n; i++){
        for(size_t j = 0; j < n; j++){
            hessian[i][j] *= 2.0/ data.size();
        }
    }

    return hessian;
}

pair<vector<double>,double> NewtonTr::optimize(const vector<array<double, 5>>& data_train, const vector<double>& initialPoint) {
   evalCount = 0;
   vector<double> beta = initialPoint;
   vector<double> grad = gradient(beta, data_train);
   vector<vector<double>> hess = hessian(beta, data_train);
   if (!isPositiveDefinite(hess)) {
       hessianAproximation(hess);
   }

    double delta_k = 0.1*norm(grad);
    double maxDelta = 1;
    double k = 0;
    vector<double> pk(beta.size(), 0.0);
    double d1 = 0.25;
    double d2 = 0.75;
    double gamma = 0.25;
    double hta = 0.25;
    
    while (norm(gradient(beta, data_train)) >= this->tolerance) {
        if (evalCount >= static_cast<size_t>(maxIterations)) {
            break;
        }
        grad = gradient(beta, data_train);
        hess = hessian(beta, data_train);
        if (!isPositiveDefinite(hess)) {
            hessianAproximation(hess);
        }

        pk = doglegDirection(grad, hess, delta_k);
 
        double f_xk = objectiveFunction(beta, data_train);
        double f_xk_pk = objectiveFunction(addVectors(beta,pk), data_train);
        double m_xk_pk = modelFunction(f_xk, grad, hess, pk);
        double m_zero = f_xk;
        double rho = (f_xk - f_xk_pk)/(m_zero- m_xk_pk);
        if(rho<d1){
            delta_k = gamma*delta_k;
        
        }else if((rho>d2) && (abs(norm(pk) - delta_k) < 1e-3)){
            delta_k = min(2*delta_k, maxDelta);
        }
       
        if(rho>hta){
            beta = addVectors(beta, pk);
            clampParameters(beta);
        }
        
        k++;
        if (norm(pk) < this->tolerance) {
            break;
        }

    }


    return {beta, static_cast<double>(evalCount)};
}
vector<double>NewtonTr::doglegDirection(const vector<double> gk, const vector<vector<double>> BMatrix, double delta_k){
    vector<vector<double>> invertBMatrix = BMatrix;
    invertMatrix(BMatrix, invertBMatrix);
    
    vector<double> pB =negateVector(matrixVectorMultiplication(invertBMatrix, gk));
    vector<double> pU(gk.size(),0.0);
    vector<double> p(gk.size(),0.0);
    double tau = 0.0;   
    double sum = 0.0;
    if(norm(pB) <= delta_k){
        p=pB;
        return p;
    }
    else{
        pU = negateVector(BFGS::scalarMultiplyVector((dotProduct(gk,gk)/dotProduct(gk,matrixVectorMultiplication(BMatrix,gk))),gk));
        if(norm(pU)>= delta_k){
            for(size_t i = 0; i < gk.size(); i++){
                p[i] = -(delta_k/norm(gk))*gk[i];
                
            }
            return p;
        }else{

            tau = solveTau(pU,pB,delta_k);
            p = addVectors(pU,BFGS::scalarMultiplyVector((tau-1),negateVector(addVectors(pB,pU))));
        }
    }
    return p;
}
vector<double> NewtonTr:: addVectors(const vector<double>& v1, const vector<double>& v2) {
    vector<double> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

void NewtonTr::hessianAproximation( vector<vector<double>>& matrix){
    double b = 1;
    double minDiagonalElement = numeric_limits<double>::max();
    
    double t;
    vector<vector<double>> B(matrix.size(), vector<double>(matrix.size(), 0));
    vector<vector<double>> I(matrix.size(), vector<double>(matrix.size(), 0));
    for (size_t i = 0; i < matrix.size(); ++i) {
        I[i][i] = 1;
    }
    for (size_t i = 0; i < matrix.size(); ++i) {
        minDiagonalElement = min(minDiagonalElement, matrix[i][i]);
    }
    if (minDiagonalElement>0) {
        t = b;
    }else{
        t = -minDiagonalElement + b;
    }

    while(true){
        for(size_t i =0; i<matrix.size(); i++){
            for(size_t j = 0; j<matrix.size(); j++){
                B[i][j] = matrix[i][j] + t*I[i][j];
            }
        }
        if(isPositiveDefinite(B)){
            matrix = B;
            break;
        }else{
           t = max(2*t, b);
        }
    }
}

bool NewtonTr::quadraticForm(const vector<double>& v, const vector<vector<double>>& H, const vector<double>& u) {
    vector<double> Hu(u.size(), 0.0);
    for (size_t i = 0; i < H.size(); ++i) {
        for (size_t j = 0; j < H[i].size(); ++j) {
            Hu[i] += H[i][j] * u[j];
        }
    }

    double result = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        result += v[i] * Hu[i];
    }

    return result > 0;

}
bool NewtonTr:: isPositiveDefinite(const vector<vector<double>>& matrix) {
    size_t n = matrix.size();
    vector<vector<double>> L(n, vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = matrix[i][j];
            for (size_t k = 0; k < j; ++k) {
                sum -= L[i][k] * L[j][k];
            }

            if (i == j) {
                if (sum <= 0.0) {
                    return false;
                }
                L[i][j] = sqrt(sum);
            } else {
                if (L[j][j] == 0.0) {
                    return false;
                }
                L[i][j] = sum / L[j][j];
            }
        }
    }
    return true;
}


void NewtonTr::forwardSubstitution(const vector<vector<double>>& L, const vector<vector<double>> I, vector<vector<double>>& Y){
    size_t n = L.size();
    Y = I; 
    for(int i =0; i<n; i++){
        for(int j=0; j<n; j++){
            for(int k = 0; k<j; k++){
                Y[j][i] -= L[j][k]*Y[k][i];
            }
            Y[j][i] /= L[j][j];
        }
    }
}
void NewtonTr::backwardSubstitution(const vector<vector<double>>& U, const vector<vector<double>> Y, vector<vector<double>>& X) {
    size_t n = U.size();
    X = Y;

    for (int i = n - 1; i >= 0; i--) {
        for (size_t j = 0; j < n; j++) {
            for (size_t k = i + 1; k < n; k++) {
                X[i][j] -= U[i][k] * X[k][j];
            }
            X[i][j] /= U[i][i];
        }
    }
}
void NewtonTr::luDecomposition(const vector<vector<double>>&matrix, vector<vector<double>>& L,vector<vector<double>>& U){
   size_t n = matrix.size();
   L = vector<vector<double>>(n, vector<double>(n, 0));
   U = vector<vector<double>>(n, vector<double>(n, 0)); 
   int i = 0, j = 0, k = 0;
    for (i = 0; i < n; i++) {
            for (k = i; k < n; k++) {
                double sum = 0;
                for (j = 0; j < i; j++) {
                    sum += L[i][j] * U[j][k];
                }
                U[i][k] = matrix[i][k] - sum;
            }

            for (k = i + 1; k < n; k++) {
                double sum = 0;
                for (j = 0; j < i; j++) {
                    sum += L[k][j] * U[j][i];
                }
                L[k][i] = (matrix[k][i] - sum) / U[i][i];
            }
            
            L[i][i] = 1;
        }

}
void NewtonTr::invertMatrix(const vector<vector<double>>& matrix, vector<vector<double>>& inverse) {
    size_t n = matrix.size();


    vector<vector<double>> L, U;
    luDecomposition(matrix, L, U);

    vector<vector<double>> I(n, vector<double>(n, 0));
    for (size_t i = 0; i < n; i++) {
        I[i][i] = 1;
    }

    vector<vector<double>> Y;
    forwardSubstitution(L, I, Y);

    backwardSubstitution(U, Y, inverse);    
}


vector<double>NewtonTr::matrixVectorMultiplication(const vector<vector<double>>& matrix, const vector<double>& vector_) {
    size_t n = matrix.size();
    vector<double> result(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i] += matrix[i][j] * vector_[j];
        }
    }
    return result;
}
vector<double>NewtonTr::negateVector(const vector<double>& vec) {
    vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = -vec[i];
    }
    return result;
}
double NewtonTr::dotProduct(const vector<double>& a, const vector<double>& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}
double NewtonTr::norm(const vector<double>& vector) {
    double result = 0.0;
    for (double d : vector) {
        result += d * d;
    }
    return sqrt(result);
}

double NewtonTr::solveTau(const vector<double> pU, const vector<double> pB, double delta_k) {
    double A = 0, B = 0, C = 0;
    size_t n = pU.size();
    double tau1 = numeric_limits<double>::infinity();
    double tau2 = numeric_limits<double>::infinity();
    vector<double> diff = addVectors(pB, negateVector(pU));
    A = dotProduct(diff,diff);
    B = 2 * dotProduct(pU, diff);
    C = dotProduct(pU,pU);

    double discriminant = (B - 2 * A) * (B - 2 * A) - 4 * A * (A - B + C);

    if (discriminant >= 0) {
        tau1 = (-(2 * A + B) + sqrt(discriminant)) / (2 * A);
        tau2 = (-(2 * A + B) - sqrt(discriminant)) / (2 * A);
    }

    double tau = numeric_limits<double>::infinity();
    if (tau1 > 0) tau = tau1;
    if (tau2 > 0 && tau2 < tau) tau = tau2;

    return (tau == numeric_limits<double>::infinity()) ? 0 : tau;
}


double NewtonTr::modelFunction(double fk, const vector<double>& g_k, const vector<vector<double>>& B_k, const vector<double>& p_k){
    double mk = fk + dotProduct(g_k,p_k) + 0.5*(dotProduct(p_k,matrixVectorMultiplication(B_k,p_k)));

    return mk;
}
