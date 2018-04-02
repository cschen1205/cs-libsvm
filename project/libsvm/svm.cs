﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PredictiveModels.Tools.Compute.SVM.libsvm
{
    public class SVM
    {
        //
        // construct and solve various formulations
        //
        public static int LIBSVM_VERSION = 320;
        public static Random rand = new Random();

        private static svm_print_interface svm_print_stdout = new svm_print_stdout();

        private static svm_print_interface svm_print_string = svm_print_stdout;

        public static void info(String s)
        {
            svm_print_string.print(s);
        }

        private static void solve_c_svc(SVMProblem prob, SVMParam param,
                        double[] alpha, Solver.SolutionInfo si,
                        double Cp, double Cn)
        {
            int l = prob.ProblemSize;
            double[] minus_ones = new double[l];
            int[] y = new int[l];

            int i;

            for (i = 0; i < l; i++)
            {
                alpha[i] = 0;
                minus_ones[i] = -1;
                if (prob.y[i] > 0) y[i] = +1; else y[i] = -1;
            }

            Solver s = new Solver();
            s.Solve(l, new SVC_Q(prob, param, y), minus_ones, y,
                alpha, Cp, Cn, param.Epsilon, si, param.UseShrinkingHeuristic);

            double sum_alpha = 0;
            for (i = 0; i < l; i++)
                sum_alpha += alpha[i];

            if (Cp == Cn)
                SVM.info("nu = " + sum_alpha / (Cp * prob.ProblemSize) + "\n");

            for (i = 0; i < l; i++)
                alpha[i] *= y[i];
        }

        private static void solve_nu_svc(SVMProblem prob, SVMParam param,
                        double[] alpha, Solver.SolutionInfo si)
        {
            int i;
            int l = prob.ProblemSize;
            double nu = param.nu;

            int[] y = new int[l];

            for (i = 0; i < l; i++)
                if (prob.y[i] > 0)
                    y[i] = +1;
                else
                    y[i] = -1;

            double sum_pos = nu * l / 2;
            double sum_neg = nu * l / 2;

            for (i = 0; i < l; i++)
                if (y[i] == +1)
                {
                    alpha[i] = Math.Min(1.0, sum_pos);
                    sum_pos -= alpha[i];
                }
                else
                {
                    alpha[i] = Math.Min(1.0, sum_neg);
                    sum_neg -= alpha[i];
                }

            double[] zeros = new double[l];

            for (i = 0; i < l; i++)
                zeros[i] = 0;

            Solver_NU s = new Solver_NU();
            s.Solve(l, new SVC_Q(prob, param, y), zeros, y,
                alpha, 1.0, 1.0, param.Epsilon, si, param.UseShrinkingHeuristic);
            double r = si.r;

            SVM.info("C = " + 1 / r + "\n");

            for (i = 0; i < l; i++)
                alpha[i] *= y[i] / r;

            si.Rho /= r;
            si.Obj /= (r * r);
            si.UpperBoundP = 1 / r;
            si.UpperBoundN = 1 / r;
        }

        private static void solve_one_class(SVMProblem prob, SVMParam param,
                        double[] alpha, Solver.SolutionInfo si)
        {
            int l = prob.ProblemSize;
            double[] zeros = new double[l];
            int[] ones = new int[l];
            int i;

            int n = (int)(param.nu * prob.ProblemSize);   // # of alpha's at upper bound

            for (i = 0; i < n; i++)
                alpha[i] = 1;
            if (n < prob.ProblemSize)
                alpha[n] = param.nu * prob.ProblemSize - n;
            for (i = n + 1; i < l; i++)
                alpha[i] = 0;

            for (i = 0; i < l; i++)
            {
                zeros[i] = 0;
                ones[i] = 1;
            }

            Solver s = new Solver();
            s.Solve(l, new ONE_CLASS_Q(prob, param), zeros, ones,
                alpha, 1.0, 1.0, param.Epsilon, si, param.UseShrinkingHeuristic);
        }

        private static void solve_epsilon_svr(SVMProblem prob, SVMParam param,
                        double[] alpha, Solver.SolutionInfo si)
        {
            int l = prob.ProblemSize;
            double[] alpha2 = new double[2 * l];
            double[] linear_term = new double[2 * l];
            int[] y = new int[2 * l];
            int i;

            for (i = 0; i < l; i++)
            {
                alpha2[i] = 0;
                linear_term[i] = param.p - prob.y[i];
                y[i] = 1;

                alpha2[i + l] = 0;
                linear_term[i + l] = param.p + prob.y[i];
                y[i + l] = -1;
            }

            Solver s = new Solver();
            s.Solve(2 * l, new SVR_Q(prob, param), linear_term, y,
                alpha2, param.C, param.C, param.Epsilon, si, param.UseShrinkingHeuristic);

            double sum_alpha = 0;
            for (i = 0; i < l; i++)
            {
                alpha[i] = alpha2[i] - alpha2[i + l];
                sum_alpha += Math.Abs(alpha[i]);
            }
            SVM.info("nu = " + sum_alpha / (param.C * l) + "\n");
        }

        private static void solve_nu_svr(SVMProblem prob, SVMParam param,
                        double[] alpha, Solver.SolutionInfo si)
        {
            int l = prob.ProblemSize;
            double C = param.C;
            double[] alpha2 = new double[2 * l];
            double[] linear_term = new double[2 * l];
            int[] y = new int[2 * l];
            int i;

            double sum = C * param.nu * l / 2;
            for (i = 0; i < l; i++)
            {
                alpha2[i] = alpha2[i + l] = Math.Min(sum, C);
                sum -= alpha2[i];

                linear_term[i] = -prob.y[i];
                y[i] = 1;

                linear_term[i + l] = prob.y[i];
                y[i + l] = -1;
            }

            Solver_NU s = new Solver_NU();
            s.Solve(2 * l, new SVR_Q(prob, param), linear_term, y,
                alpha2, C, C, param.Epsilon, si, param.UseShrinkingHeuristic);

            SVM.info("epsilon = " + (-si.r) + "\n");

            for (i = 0; i < l; i++)
                alpha[i] = alpha2[i] - alpha2[i + l];
        }

        //
        // decision_function
        //
        class decision_function
        {
            public double[] alpha;
            public double rho;
        }

        static decision_function svm_train_one(
            SVMProblem prob, SVMParam param,
            double Cp, double Cn)
        {
            double[] alpha = new double[prob.ProblemSize];
            Solver.SolutionInfo si = new Solver.SolutionInfo();
            switch (param.SVMType)
            {
                case SVMParam.SVM_TYPE_C_SVC:
                    solve_c_svc(prob, param, alpha, si, Cp, Cn);
                    break;
                case SVMParam.SVM_TYPE_NU_SVC:
                    solve_nu_svc(prob, param, alpha, si);
                    break;
                case SVMParam.SVM_TYPE_ONE_CLASS:
                    solve_one_class(prob, param, alpha, si);
                    break;
                case SVMParam.SVM_TYPE_EPSILON_SVR:
                    solve_epsilon_svr(prob, param, alpha, si);
                    break;
                case SVMParam.SVM_TYPE_NU_SVR:
                    solve_nu_svr(prob, param, alpha, si);
                    break;
            }

            SVM.info("obj = " + si.Obj + ", rho = " + si.Rho + "\n");

            // output SVs

            int nSV = 0;
            int nBSV = 0;
            for (int i = 0; i < prob.ProblemSize; i++)
            {
                if (Math.Abs(alpha[i]) > 0)
                {
                    ++nSV;
                    if (prob.y[i] > 0)
                    {
                        if (Math.Abs(alpha[i]) >= si.UpperBoundP)
                            ++nBSV;
                    }
                    else
                    {
                        if (Math.Abs(alpha[i]) >= si.UpperBoundN)
                            ++nBSV;
                    }
                }
            }

            SVM.info("nSV = " + nSV + ", nBSV = " + nBSV + "\n");

            decision_function f = new decision_function();
            f.alpha = alpha;
            f.rho = si.Rho;
            return f;
        }

        // Platt's binary SVM Probablistic Output: an improvement from Lin et al.
        private static void sigmoid_train(int l, double[] dec_values, double[] labels,
                      double[] probAB)
        {
            double A, B;
            double prior1 = 0, prior0 = 0;
            int i;

            for (i = 0; i < l; i++)
                if (labels[i] > 0) prior1 += 1;
                else prior0 += 1;

            int max_iter = 100; // Maximal number of iterations
            double min_step = 1e-10;    // Minimal step taken in line search
            double sigma = 1e-12;   // For numerically strict PD of Hessian
            double eps = 1e-5;
            double hiTarget = (prior1 + 1.0) / (prior1 + 2.0);
            double loTarget = 1 / (prior0 + 2.0);
            double[] t = new double[l];
            double fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize;
            double newA, newB, newf, d1, d2;
            int iter;

            // Initial Point and Initial Fun Value
            A = 0.0; B = Math.Log((prior0 + 1.0) / (prior1 + 1.0));
            double fval = 0.0;

            for (i = 0; i < l; i++)
            {
                if (labels[i] > 0) t[i] = hiTarget;
                else t[i] = loTarget;
                fApB = dec_values[i] * A + B;
                if (fApB >= 0)
                    fval += t[i] * fApB + Math.Log(1 + Math.Exp(-fApB));
                else
                    fval += (t[i] - 1) * fApB + Math.Log(1 + Math.Exp(fApB));
            }
            for (iter = 0; iter < max_iter; iter++)
            {
                // Update Gradient and Hessian (use H' = H + sigma I)
                h11 = sigma; // numerically ensures strict PD
                h22 = sigma;
                h21 = 0.0; g1 = 0.0; g2 = 0.0;
                for (i = 0; i < l; i++)
                {
                    fApB = dec_values[i] * A + B;
                    if (fApB >= 0)
                    {
                        p = Math.Exp(-fApB) / (1.0 + Math.Exp(-fApB));
                        q = 1.0 / (1.0 + Math.Exp(-fApB));
                    }
                    else
                    {
                        p = 1.0 / (1.0 + Math.Exp(fApB));
                        q = Math.Exp(fApB) / (1.0 + Math.Exp(fApB));
                    }
                    d2 = p * q;
                    h11 += dec_values[i] * dec_values[i] * d2;
                    h22 += d2;
                    h21 += dec_values[i] * d2;
                    d1 = t[i] - p;
                    g1 += dec_values[i] * d1;
                    g2 += d1;
                }

                // Stopping Criteria
                if (Math.Abs(g1) < eps && Math.Abs(g2) < eps)
                    break;

                // Finding Newton direction: -inv(H') * g
                det = h11 * h22 - h21 * h21;
                dA = -(h22 * g1 - h21 * g2) / det;
                dB = -(-h21 * g1 + h11 * g2) / det;
                gd = g1 * dA + g2 * dB;


                stepsize = 1;       // Line Search
                while (stepsize >= min_step)
                {
                    newA = A + stepsize * dA;
                    newB = B + stepsize * dB;

                    // New function value
                    newf = 0.0;
                    for (i = 0; i < l; i++)
                    {
                        fApB = dec_values[i] * newA + newB;
                        if (fApB >= 0)
                            newf += t[i] * fApB + Math.Log(1 + Math.Exp(-fApB));
                        else
                            newf += (t[i] - 1) * fApB + Math.Log(1 + Math.Exp(fApB));
                    }
                    // Check sufficient decrease
                    if (newf < fval + 0.0001 * stepsize * gd)
                    {
                        A = newA; B = newB; fval = newf;
                        break;
                    }
                    else
                        stepsize = stepsize / 2.0;
                }

                if (stepsize < min_step)
                {
                    SVM.info("Line search fails in two-class probability estimates\n");
                    break;
                }
            }

            if (iter >= max_iter)
                SVM.info("Reaching maximal iterations in two-class probability estimates\n");
            probAB[0] = A; probAB[1] = B;
        }

        private static double sigmoid_predict(double decision_value, double A, double B)
        {
            double fApB = decision_value * A + B;
            if (fApB >= 0)
                return Math.Exp(-fApB) / (1.0 + Math.Exp(-fApB));
            else
                return 1.0 / (1 + Math.Exp(fApB));
        }

        // Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
        private static void multiclass_probability(int k, double[][] r, double[] p)
        {
            int t, j;
            int iter = 0, max_iter = Math.Max(100, k);
            double[][] Q = new double[k][];
            for(int i=0; i < k; ++i)
            {
                Q[i] = new double[k];
            }
            
            double[] Qp = new double[k];
            double pQp, eps = 0.005 / k;

            for (t = 0; t < k; t++)
            {
                p[t] = 1.0 / k;  // Valid if k = 1
                Q[t][t] = 0;
                for (j = 0; j < t; j++)
                {
                    Q[t][t] += r[j][t] * r[j][t];
                    Q[t][j] = Q[j][t];
                }
                for (j = t + 1; j < k; j++)
                {
                    Q[t][t] += r[j][t] * r[j][t];
                    Q[t][j] = -r[j][t] * r[t][j];
                }
            }
            for (iter = 0; iter < max_iter; iter++)
            {
                // stopping condition, recalculate QP,pQP for numerical accuracy
                pQp = 0;
                for (t = 0; t < k; t++)
                {
                    Qp[t] = 0;
                    for (j = 0; j < k; j++)
                        Qp[t] += Q[t][j] * p[j];
                    pQp += p[t] * Qp[t];
                }
                double max_error = 0;
                for (t = 0; t < k; t++)
                {
                    double error = Math.Abs(Qp[t] - pQp);
                    if (error > max_error)
                        max_error = error;
                }
                if (max_error < eps) break;

                for (t = 0; t < k; t++)
                {
                    double diff = (-Qp[t] + pQp) / Q[t][t];
                    p[t] += diff;
                    pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff) / (1 + diff);
                    for (j = 0; j < k; j++)
                    {
                        Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff);
                        p[j] /= (1 + diff);
                    }
                }
            }
            if (iter >= max_iter)
                SVM.info("Exceeds max_iter in multiclass_prob\n");
        }

        // Cross-validation decision values for probability estimates
        private static void svm_binary_svc_probability(SVMProblem prob, SVMParam param, double Cp, double Cn, double[] probAB, Func<int, bool> shouldTerminate)
        {
            int i;
            int nr_fold = 5;
            int[] perm = new int[prob.ProblemSize];
            double[] dec_values = new double[prob.ProblemSize];

            // naive shuffle
            for (i = 0; i < prob.ProblemSize; i++) perm[i] = i;
            for (i = 0; i < prob.ProblemSize; i++)
            {
                int j = i + rand.Next(prob.ProblemSize - i);
                do { int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp; } while (false);
            }
            for (i = 0; i < nr_fold; i++)
            {
                int begin = i * prob.ProblemSize / nr_fold;
                int end = (i + 1) * prob.ProblemSize / nr_fold;
                int j, k;
                SVMProblem subprob = new SVMProblem();

                subprob.ProblemSize = prob.ProblemSize - (end - begin);
                subprob.x = new SVMNode[subprob.ProblemSize][];
                subprob.y = new double[subprob.ProblemSize];

                k = 0;
                for (j = 0; j < begin; j++)
                {
                    subprob.x[k] = prob.x[perm[j]];
                    subprob.y[k] = prob.y[perm[j]];
                    ++k;
                }
                for (j = end; j < prob.ProblemSize; j++)
                {
                    subprob.x[k] = prob.x[perm[j]];
                    subprob.y[k] = prob.y[perm[j]];
                    ++k;
                }
                int p_count = 0, n_count = 0;
                for (j = 0; j < k; j++)
                    if (subprob.y[j] > 0)
                        p_count++;
                    else
                        n_count++;

                if (p_count == 0 && n_count == 0)
                    for (j = begin; j < end; j++)
                        dec_values[perm[j]] = 0;
                else if (p_count > 0 && n_count == 0)
                    for (j = begin; j < end; j++)
                        dec_values[perm[j]] = 1;
                else if (p_count == 0 && n_count > 0)
                    for (j = begin; j < end; j++)
                        dec_values[perm[j]] = -1;
                else
                {
                    SVMParam subparam = (SVMParam)param.Clone();
                    subparam.DoProbabilityEstimate = false;
                    subparam.C = 1.0;
                    subparam.NumberWeight = 2;
                    subparam.WeightLabel = new int[2];
                    subparam.Weight = new double[2];
                    subparam.WeightLabel[0] = +1;
                    subparam.WeightLabel[1] = -1;
                    subparam.Weight[0] = Cp;
                    subparam.Weight[1] = Cn;
                    SVMModel submodel = svm_train(subprob, subparam, shouldTerminate);
                    for (j = begin; j < end; j++)
                    {
                        double[] dec_value = new double[1];
                        svm_predict_values(submodel, prob.x[perm[j]], dec_value);
                        dec_values[perm[j]] = dec_value[0];
                        // ensure +1 -1 order; reason not using CV subroutine
                        dec_values[perm[j]] *= submodel.Label[0];
                    }
                }
            }
            sigmoid_train(prob.ProblemSize, dec_values, prob.y, probAB);
        }

        // Return parameter of a Laplace distribution 
        private static double svm_svr_probability(SVMProblem prob, SVMParam param, Func<int, bool> shouldTerminate)
        {
            int i;
            int nr_fold = 5;
            double[] ymv = new double[prob.ProblemSize];
            double mae = 0;

            SVMParam newparam = (SVMParam)param.Clone();
            newparam.DoProbabilityEstimate = false;
            svm_cross_validation(prob, newparam, nr_fold, ymv, shouldTerminate);
            for (i = 0; i < prob.ProblemSize; i++)
            {
                ymv[i] = prob.y[i] - ymv[i];
                mae += Math.Abs(ymv[i]);
            }
            mae /= prob.ProblemSize;
            double std = Math.Sqrt(2 * mae * mae);
            int count = 0;
            mae = 0;
            for (i = 0; i < prob.ProblemSize; i++)
                if (Math.Abs(ymv[i]) > 5 * std)
                    count = count + 1;
                else
                    mae += Math.Abs(ymv[i]);
            mae /= (prob.ProblemSize - count);
            SVM.info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=" + mae + "\n");
            return mae;
        }

        // label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
        // perm, length l, must be allocated before calling this subroutine
        private static void svm_group_classes(SVMProblem prob, int[] nr_class_ret, int[][] label_ret, int[][] start_ret, int[][] count_ret, int[] perm)
        {
            int l = prob.ProblemSize;
            int max_nr_class = 16;
            int nr_class = 0;
            int[] label = new int[max_nr_class];
            int[] count = new int[max_nr_class];
            int[] data_label = new int[l];
            int i;

            for (i = 0; i < l; i++)
            {
                int this_label = (int)(prob.y[i]);
                int j;
                for (j = 0; j < nr_class; j++)
                {
                    if (this_label == label[j])
                    {
                        ++count[j];
                        break;
                    }
                }
                data_label[i] = j;
                if (j == nr_class)
                {
                    if (nr_class == max_nr_class)
                    {
                        max_nr_class *= 2;
                        int[] new_data = new int[max_nr_class];
                        
                        for(int mindex = 0; mindex < label.Length; ++mindex)
                        {
                            new_data[mindex] = label[mindex];
                        }

                        label = new_data;
                        new_data = new int[max_nr_class];
                        
                        for (int mindex = 0; mindex < count.Length; ++mindex)
                        {
                            new_data[mindex] = count[mindex];
                        }

                        count = new_data;
                    }
                    label[nr_class] = this_label;
                    count[nr_class] = 1;
                    ++nr_class;
                }
            }

            //
            // Labels are ordered by their first occurrence in the training set. 
            // However, for two-class sets with -1/+1 labels and -1 appears first, 
            // we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
            //
            if (nr_class == 2 && label[0] == -1 && label[1] == +1)
            {
                do { int tmp = label[0]; label[0] = label[1]; label[1] = tmp; } while (false);
                do { int tmp = count[0]; count[0] = count[1]; count[1] = tmp; } while (false);
                for (i = 0; i < l; i++)
                {
                    if (data_label[i] == 0)
                        data_label[i] = 1;
                    else
                        data_label[i] = 0;
                }
            }

            int[] start = new int[nr_class];
            start[0] = 0;
            for (i = 1; i < nr_class; i++)
                start[i] = start[i - 1] + count[i - 1];
            for (i = 0; i < l; i++)
            {
                perm[start[data_label[i]]] = i;
                ++start[data_label[i]];
            }
            start[0] = 0;
            for (i = 1; i < nr_class; i++)
                start[i] = start[i - 1] + count[i - 1];

            nr_class_ret[0] = nr_class;
            label_ret[0] = label;
            start_ret[0] = start;
            count_ret[0] = count;
        }

        //
        // Interface functions
        //
        public static SVMModel svm_train(SVMProblem prob, SVMParam param, Func<int, bool> shouldTerminate)
        {
            SVMModel model = new SVMModel();
            model.Param = param;

            if (param.SVMType == SVMParam.SVM_TYPE_ONE_CLASS ||
               param.SVMType == SVMParam.SVM_TYPE_EPSILON_SVR ||
               param.SVMType == SVMParam.SVM_TYPE_NU_SVR)
            {
                // regression or one-class-svm
                model.NumberClass = 2;
                model.Label = null;
                model.NumberSV4EachClass = null;
                model.ProbA = null; model.ProbB = null;
                model.CoefSV = new double[1][];

                if (param.DoProbabilityEstimate &&
                   (param.SVMType == SVMParam.SVM_TYPE_EPSILON_SVR ||
                    param.SVMType == SVMParam.SVM_TYPE_NU_SVR))
                {
                    model.ProbA = new double[1];
                    model.ProbA[0] = svm_svr_probability(prob, param, shouldTerminate);
                }

                decision_function f = svm_train_one(prob, param, 0, 0);
                model.Rho = new double[1];
                model.Rho[0] = f.rho;

                int nSV = 0;
                int i;
                for (i = 0; i < prob.ProblemSize; i++)
                    if (Math.Abs(f.alpha[i]) > 0) ++nSV;
                model.TotalNumberSV = nSV;
                model.SV = new SVMNode[nSV][];
                model.CoefSV[0] = new double[nSV];
                model.IndicesSV = new int[nSV];
                int j = 0;
                for (i = 0; i < prob.ProblemSize; i++)
                    if (Math.Abs(f.alpha[i]) > 0)
                    {
                        model.SV[j] = prob.x[i];
                        model.CoefSV[0][j] = f.alpha[i];
                        model.IndicesSV[j] = i + 1;
                        ++j;
                    }
            }
            else
            {
                // classification
                int l = prob.ProblemSize;
                int[] tmp_nr_class = new int[1];
                int[][] tmp_label = new int[1][];
                int[][] tmp_start = new int[1][];
                int[][] tmp_count = new int[1][];
                int[] perm = new int[l];

                // group training data of the same class
                svm_group_classes(prob, tmp_nr_class, tmp_label, tmp_start, tmp_count, perm);
                int nr_class = tmp_nr_class[0];
                int[] label = tmp_label[0];
                int[] start = tmp_start[0];
                int[] count = tmp_count[0];

                if (nr_class == 1)
                    SVM.info("WARNING: training data in only one class. See README for details.\n");

                SVMNode[][] x = new SVMNode[l][];
                int i;
                for (i = 0; i < l; i++)
                    x[i] = prob.x[perm[i]];

                // calculate weighted C

                double[] weighted_C = new double[nr_class];
                for (i = 0; i < nr_class; i++)
                    weighted_C[i] = param.C;
                for (i = 0; i < param.NumberWeight; i++)
                {
                    int j;
                    for (j = 0; j < nr_class; j++)
                        if (param.WeightLabel[i] == label[j])
                            break;
                    if (j == nr_class)
                        Console.WriteLine("WARNING: class label " + param.WeightLabel[i] + " specified in weight is not found\n");
                    else
                        weighted_C[j] *= param.Weight[i];
                }

                // train k*(k-1)/2 models

                bool[] nonzero = new bool[l];
                for (i = 0; i < l; i++)
                    nonzero[i] = false;
                decision_function[] f = new decision_function[nr_class * (nr_class - 1) / 2];

                double[] probA = null, probB = null;
                if (param.DoProbabilityEstimate)
                {
                    probA = new double[nr_class * (nr_class - 1) / 2];
                    probB = new double[nr_class * (nr_class - 1) / 2];
                }

                int p = 0;
                for (i = 0; i < nr_class; i++)
                    for (int j = i + 1; j < nr_class; j++)
                    {
                        SVMProblem sub_prob = new SVMProblem();
                        int si = start[i], sj = start[j];
                        int ci = count[i], cj = count[j];
                        sub_prob.ProblemSize = ci + cj;
                        sub_prob.x = new SVMNode[sub_prob.ProblemSize][];
                        sub_prob.y = new double[sub_prob.ProblemSize];
                        int k;
                        for (k = 0; k < ci; k++)
                        {
                            sub_prob.x[k] = x[si + k];
                            sub_prob.y[k] = +1;
                        }
                        for (k = 0; k < cj; k++)
                        {
                            sub_prob.x[ci + k] = x[sj + k];
                            sub_prob.y[ci + k] = -1;
                        }

                        if (param.DoProbabilityEstimate)
                        {
                            double[] probAB = new double[2];
                            svm_binary_svc_probability(sub_prob, param, weighted_C[i], weighted_C[j], probAB, shouldTerminate);
                            probA[p] = probAB[0];
                            probB[p] = probAB[1];
                        }

                        f[p] = svm_train_one(sub_prob, param, weighted_C[i], weighted_C[j]);
                        for (k = 0; k < ci; k++)
                            if (!nonzero[si + k] && Math.Abs(f[p].alpha[k]) > 0)
                                nonzero[si + k] = true;
                        for (k = 0; k < cj; k++)
                            if (!nonzero[sj + k] && Math.Abs(f[p].alpha[ci + k]) > 0)
                                nonzero[sj + k] = true;
                        ++p;
                    }

                // build output

                model.NumberClass = nr_class;

                model.Label = new int[nr_class];
                for (i = 0; i < nr_class; i++)
                    model.Label[i] = label[i];

                model.Rho = new double[nr_class * (nr_class - 1) / 2];
                for (i = 0; i < nr_class * (nr_class - 1) / 2; i++)
                    model.Rho[i] = f[i].rho;

                if (param.DoProbabilityEstimate)
                {
                    model.ProbA = new double[nr_class * (nr_class - 1) / 2];
                    model.ProbB = new double[nr_class * (nr_class - 1) / 2];
                    for (i = 0; i < nr_class * (nr_class - 1) / 2; i++)
                    {
                        model.ProbA[i] = probA[i];
                        model.ProbB[i] = probB[i];
                    }
                }
                else
                {
                    model.ProbA = null;
                    model.ProbB = null;
                }

                int nnz = 0;
                int[] nz_count = new int[nr_class];
                model.NumberSV4EachClass = new int[nr_class];
                for (i = 0; i < nr_class; i++)
                {
                    int nSV = 0;
                    for (int j = 0; j < count[i]; j++)
                        if (nonzero[start[i] + j])
                        {
                            ++nSV;
                            ++nnz;
                        }
                    model.NumberSV4EachClass[i] = nSV;
                    nz_count[i] = nSV;
                }

                SVM.info("Total nSV = " + nnz + "\n");

                model.TotalNumberSV = nnz;
                model.SV = new SVMNode[nnz][];
                model.IndicesSV = new int[nnz];
                p = 0;
                for (i = 0; i < l; i++)
                    if (nonzero[i])
                    {
                        model.SV[p] = x[i];
                        model.IndicesSV[p++] = perm[i] + 1;
                    }

                int[] nz_start = new int[nr_class];
                nz_start[0] = 0;
                for (i = 1; i < nr_class; i++)
                    nz_start[i] = nz_start[i - 1] + nz_count[i - 1];

                model.CoefSV = new double[nr_class - 1][];
                for (i = 0; i < nr_class - 1; i++)
                    model.CoefSV[i] = new double[nnz];

                p = 0;
                for (i = 0; i < nr_class; i++)
                    for (int j = i + 1; j < nr_class; j++)
                    {
                        // classifier (i,j): coefficients with
                        // i are in sv_coef[j-1][nz_start[i]...],
                        // j are in sv_coef[i][nz_start[j]...]

                        int si = start[i];
                        int sj = start[j];
                        int ci = count[i];
                        int cj = count[j];

                        int q = nz_start[i];
                        int k;
                        for (k = 0; k < ci; k++)
                            if (nonzero[si + k])
                                model.CoefSV[j - 1][q++] = f[p].alpha[k];
                        q = nz_start[j];
                        for (k = 0; k < cj; k++)
                            if (nonzero[sj + k])
                                model.CoefSV[i][q++] = f[p].alpha[ci + k];
                        ++p;
                    }
            }
            return model;
        }

        // Stratified cross validation
        public static void svm_cross_validation(SVMProblem prob, SVMParam param, int nr_fold, double[] target, Func<int, bool> shouldTerminate)
        {
            int i;
            int[] fold_start = new int[nr_fold + 1];
            int l = prob.ProblemSize;
            int[] perm = new int[l];

            // stratified cv may not give leave-one-out rate
            // Each class to l folds -> some folds may have zero elements
            if ((param.SVMType == SVMParam.SVM_TYPE_C_SVC ||
                param.SVMType == SVMParam.SVM_TYPE_NU_SVC) && nr_fold < l)
            {
                int[] tmp_nr_class = new int[1];
                int[][] tmp_label = new int[1][];
                int[][] tmp_start = new int[1][];
                int[][] tmp_count = new int[1][];

                svm_group_classes(prob, tmp_nr_class, tmp_label, tmp_start, tmp_count, perm);

                int nr_class = tmp_nr_class[0];
                int[] start = tmp_start[0];
                int[] count = tmp_count[0];

                // naive shuffle and then data grouped by fold using the array perm
                int[] fold_count = new int[nr_fold];
                int c;
                int[] index = new int[l];
                for (i = 0; i < l; i++)
                    index[i] = perm[i];
                for (c = 0; c < nr_class; c++)
                    for (i = 0; i < count[c]; i++)
                    {
                        int j = i + rand.Next(count[c] - i);
                        do { int tmp = index[start[c] + j]; index[start[c] + j] = index[start[c] + i]; index[start[c] + i] = tmp; } while (false);
                    }
                for (i = 0; i < nr_fold; i++)
                {
                    fold_count[i] = 0;
                    for (c = 0; c < nr_class; c++)
                        fold_count[i] += (i + 1) * count[c] / nr_fold - i * count[c] / nr_fold;
                }
                fold_start[0] = 0;
                for (i = 1; i <= nr_fold; i++)
                    fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
                for (c = 0; c < nr_class; c++)
                    for (i = 0; i < nr_fold; i++)
                    {
                        int begin = start[c] + i * count[c] / nr_fold;
                        int end = start[c] + (i + 1) * count[c] / nr_fold;
                        for (int j = begin; j < end; j++)
                        {
                            perm[fold_start[i]] = index[j];
                            fold_start[i]++;
                        }
                    }
                fold_start[0] = 0;
                for (i = 1; i <= nr_fold; i++)
                    fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
            }
            else
            {
                for (i = 0; i < l; i++) perm[i] = i;
                for (i = 0; i < l; i++)
                {
                    int j = i + rand.Next(l - i);
                    do { int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp; } while (false);
                }
                for (i = 0; i <= nr_fold; i++)
                    fold_start[i] = i * l / nr_fold;
            }

            for (i = 0; i < nr_fold; i++)
            {
                int begin = fold_start[i];
                int end = fold_start[i + 1];
                int j, k;
                SVMProblem subprob = new SVMProblem();

                subprob.ProblemSize = l - (end - begin);
                subprob.x = new SVMNode[subprob.ProblemSize][];
                subprob.y = new double[subprob.ProblemSize];

                k = 0;
                for (j = 0; j < begin; j++)
                {
                    subprob.x[k] = prob.x[perm[j]];
                    subprob.y[k] = prob.y[perm[j]];
                    ++k;
                }
                for (j = end; j < l; j++)
                {
                    subprob.x[k] = prob.x[perm[j]];
                    subprob.y[k] = prob.y[perm[j]];
                    ++k;
                }
                SVMModel submodel = svm_train(subprob, param, shouldTerminate);
                if (param.DoProbabilityEstimate &&
                   (param.SVMType == SVMParam.SVM_TYPE_C_SVC ||
                    param.SVMType == SVMParam.SVM_TYPE_NU_SVC))
                {
                    double[] prob_estimates = new double[svm_get_nr_class(submodel)];
                    for (j = begin; j < end; j++)
                        target[perm[j]] = svm_predict_probability(submodel, prob.x[perm[j]], prob_estimates);
                }
                else
                    for (j = begin; j < end; j++)
                        target[perm[j]] = svm_predict(submodel, prob.x[perm[j]]);
            }
        }

        public static int svm_get_svm_type(SVMModel model)
        {
            return model.Param.SVMType;
        }

        public static int svm_get_nr_class(SVMModel model)
        {
            return model.NumberClass;
        }

        public static void svm_get_labels(SVMModel model, int[] label)
        {
            if (model.Label != null)
                for (int i = 0; i < model.NumberClass; i++)
                    label[i] = model.Label[i];
        }

        public static void svm_get_sv_indices(SVMModel model, int[] indices)
        {
            if (model.IndicesSV != null)
                for (int i = 0; i < model.TotalNumberSV; i++)
                    indices[i] = model.IndicesSV[i];
        }

        public static int svm_get_nr_sv(SVMModel model)
        {
            return model.TotalNumberSV;
        }

        public static double svm_get_svr_probability(SVMModel model)
        {
            if ((model.Param.SVMType == SVMParam.SVM_TYPE_EPSILON_SVR || model.Param.SVMType == SVMParam.SVM_TYPE_NU_SVR) &&
                model.ProbA != null)
                return model.ProbA[0];
            else
            {
                Console.WriteLine("Model doesn't contain information for SVR probability inference\n");
                return 0;
            }
        }

        public static double svm_predict_values(SVMModel model, SVMNode[] x, double[] dec_values)
        {
            int i;
            if (model.Param.SVMType == SVMParam.SVM_TYPE_ONE_CLASS ||
               model.Param.SVMType == SVMParam.SVM_TYPE_EPSILON_SVR ||
               model.Param.SVMType == SVMParam.SVM_TYPE_NU_SVR)
            {
                double[] sv_coef = model.CoefSV[0];
                double sum = 0;
                for (i = 0; i < model.TotalNumberSV; i++)
                    sum += sv_coef[i] * Kernel.k_function(x, model.SV[i], model.Param);
                sum -= model.Rho[0];
                dec_values[0] = sum;

                if (model.Param.SVMType == SVMParam.SVM_TYPE_ONE_CLASS)
                    return (sum > 0) ? 1 : -1;
                else
                    return sum;
            }
            else
            {
                int nr_class = model.NumberClass;
                int l = model.TotalNumberSV;

                double[] kvalue = new double[l];
                for (i = 0; i < l; i++)
                    kvalue[i] = Kernel.k_function(x, model.SV[i], model.Param);

                int[] start = new int[nr_class];
                start[0] = 0;
                for (i = 1; i < nr_class; i++)
                    start[i] = start[i - 1] + model.NumberSV4EachClass[i - 1];

                int[] vote = new int[nr_class];
                for (i = 0; i < nr_class; i++)
                    vote[i] = 0;

                int p = 0;
                for (i = 0; i < nr_class; i++)
                    for (int j = i + 1; j < nr_class; j++)
                    {
                        double sum = 0;
                        int si = start[i];
                        int sj = start[j];
                        int ci = model.NumberSV4EachClass[i];
                        int cj = model.NumberSV4EachClass[j];

                        int k;
                        double[] coef1 = model.CoefSV[j - 1];
                        double[] coef2 = model.CoefSV[i];
                        for (k = 0; k < ci; k++)
                            sum += coef1[si + k] * kvalue[si + k];
                        for (k = 0; k < cj; k++)
                            sum += coef2[sj + k] * kvalue[sj + k];
                        sum -= model.Rho[p];
                        dec_values[p] = sum;

                        if (dec_values[p] > 0)
                            ++vote[i];
                        else
                            ++vote[j];
                        p++;
                    }

                int vote_max_idx = 0;
                for (i = 1; i < nr_class; i++)
                    if (vote[i] > vote[vote_max_idx])
                        vote_max_idx = i;

                return model.Label[vote_max_idx];
            }
        }

        public static double svm_predict(SVMModel model, SVMNode[] x)
        {
            int nr_class = model.NumberClass;
            double[] dec_values;
            if (model.Param.SVMType == SVMParam.SVM_TYPE_ONE_CLASS ||
                    model.Param.SVMType == SVMParam.SVM_TYPE_EPSILON_SVR ||
                    model.Param.SVMType == SVMParam.SVM_TYPE_NU_SVR)
                dec_values = new double[1];
            else
                dec_values = new double[nr_class * (nr_class - 1) / 2];
            double pred_result = svm_predict_values(model, x, dec_values);
            return pred_result;
        }

        public static double svm_predict_probability(SVMModel model, SVMNode[] x, double[] prob_estimates)
        {
            if ((model.Param.SVMType == SVMParam.SVM_TYPE_C_SVC || model.Param.SVMType == SVMParam.SVM_TYPE_NU_SVC) &&
                model.ProbA != null && model.ProbB != null)
            {
                int i;
                int nr_class = model.NumberClass;
                double[] dec_values = new double[nr_class * (nr_class - 1) / 2];
                svm_predict_values(model, x, dec_values);

                double min_prob = 1e-7;
                double[][] pairwise_prob = new double[nr_class][];
                for(int mindex = 0; mindex < pairwise_prob.Length; ++mindex)
                {
                    pairwise_prob[mindex] = new double[nr_class];
                }

                int k = 0;
                for (i = 0; i < nr_class; i++)
                    for (int j = i + 1; j < nr_class; j++)
                    {
                        pairwise_prob[i][j] = Math.Min(Math.Max(sigmoid_predict(dec_values[k], model.ProbA[k], model.ProbB[k]), min_prob), 1 - min_prob);
                        pairwise_prob[j][i] = 1 - pairwise_prob[i][j];
                        k++;
                    }
                multiclass_probability(nr_class, pairwise_prob, prob_estimates);

                int prob_max_idx = 0;
                for (i = 1; i < nr_class; i++)
                    if (prob_estimates[i] > prob_estimates[prob_max_idx])
                        prob_max_idx = i;
                return model.Label[prob_max_idx];
            }
            else
                return svm_predict(model, x);
        }

        static String[] svm_type_table =
	    {
		    "c_svc","nu_svc","one_class","epsilon_svr","nu_svr",
	    };

        static String[] kernel_type_table =
	    {
		    "linear","polynomial","rbf","sigmoid","precomputed"
	    };

	    

	    private static double atof(String s)
        {
            return double.Parse(s);
        }

        private static int atoi(String s)
        {
            return int.Parse(s);
        }


	    public static String svm_check_parameter(SVMProblem prob, SVMParam param)
        {
            // svm_type

            int svm_type = param.SVMType;
            if (svm_type != SVMParam.SVM_TYPE_C_SVC &&
               svm_type != SVMParam.SVM_TYPE_NU_SVC &&
               svm_type != SVMParam.SVM_TYPE_ONE_CLASS &&
               svm_type != SVMParam.SVM_TYPE_EPSILON_SVR &&
               svm_type != SVMParam.SVM_TYPE_NU_SVR)
                return "unknown svm type";

            // kernel_type, degree

            int kernel_type = param.KernelType;
            if (kernel_type != SVMParam.KERNEL_TYPE_LINEAR &&
               kernel_type != SVMParam.KERNEL_TYPE_POLY &&
               kernel_type != SVMParam.KERNEL_TYPE_RBF &&
               kernel_type != SVMParam.KERNEL_TYPE_SIGMOID &&
               kernel_type != SVMParam.KERNEL_TYPE_PRECOMPUTED)
                return "unknown kernel type";

            if (param.Gamma < 0)
                return "gamma < 0";

            if (param.Degree < 0)
                return "degree of polynomial kernel < 0";

            // cache_size,eps,C,nu,p,shrinking

            if (param.CacheSizeInMB <= 0)
                return "cache_size <= 0";

            if (param.Epsilon <= 0)
                return "eps <= 0";

            if (svm_type == SVMParam.SVM_TYPE_C_SVC ||
               svm_type == SVMParam.SVM_TYPE_EPSILON_SVR ||
               svm_type == SVMParam.SVM_TYPE_NU_SVR)
                if (param.C <= 0)
                    return "C <= 0";

            if (svm_type == SVMParam.SVM_TYPE_NU_SVC ||
               svm_type == SVMParam.SVM_TYPE_ONE_CLASS ||
               svm_type == SVMParam.SVM_TYPE_NU_SVR)
                if (param.nu <= 0 || param.nu > 1)
                    return "nu <= 0 or nu > 1";

            if (svm_type == SVMParam.SVM_TYPE_EPSILON_SVR)
                if (param.p < 0)
                    return "p < 0";

            
            if (param.DoProbabilityEstimate &&
               svm_type == SVMParam.SVM_TYPE_ONE_CLASS)
                return "one-class SVM probability output not supported yet";

            // check whether nu-svc is feasible

            if (svm_type == SVMParam.SVM_TYPE_NU_SVC)
            {
                int l = prob.ProblemSize;
                int max_nr_class = 16;
                int nr_class = 0;
                int[] label = new int[max_nr_class];
                int[] count = new int[max_nr_class];

                int i;
                for (i = 0; i < l; i++)
                {
                    int this_label = (int)prob.y[i];
                    int j;
                    for (j = 0; j < nr_class; j++)
                        if (this_label == label[j])
                        {
                            ++count[j];
                            break;
                        }

                    if (j == nr_class)
                    {
                        if (nr_class == max_nr_class)
                        {
                            max_nr_class *= 2;
                            int[] new_data = new int[max_nr_class];

                            for(int mindex = 0; mindex < label.Length; ++mindex)
                            {
                                new_data[mindex] = label[mindex];
                            }
                            
                            label = new_data;

                            new_data = new int[max_nr_class];
                            
                            for (int mindex = 0; mindex < count.Length; ++mindex)
                            {
                                new_data[mindex] = count[mindex];
                            }

                            count = new_data;
                        }
                        label[nr_class] = this_label;
                        count[nr_class] = 1;
                        ++nr_class;
                    }
                }

                for (i = 0; i < nr_class; i++)
                {
                    int n1 = count[i];
                    for (int j = i + 1; j < nr_class; j++)
                    {
                        int n2 = count[j];
                        if (param.nu * (n1 + n2) / 2 > Math.Min(n1, n2))
                            return "specified nu is infeasible";
                    }
                }
            }

            return null;
        }

        public static int svm_check_probability_model(SVMModel model)
        {
            if (((model.Param.SVMType == SVMParam.SVM_TYPE_C_SVC || model.Param.SVMType == SVMParam.SVM_TYPE_NU_SVC) &&
            model.ProbA != null && model.ProbB != null) ||
            ((model.Param.SVMType == SVMParam.SVM_TYPE_EPSILON_SVR || model.Param.SVMType == SVMParam.SVM_TYPE_NU_SVR) &&
             model.ProbA != null))
                return 1;
            else
                return 0;
        }

        public static void svm_set_print_string_function(svm_print_interface print_func)
        {
            if (print_func == null)
                svm_print_string = svm_print_stdout;
            else
                svm_print_string = print_func;
        }
    }
}
