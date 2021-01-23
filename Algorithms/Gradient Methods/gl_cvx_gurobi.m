function [x,iter,out] = gl_cvx_gurobi(x0,A,b,mu,varargin)
cvx_solver gurobi;

m = size(A);
m = m(1);
n = size(A);
n = n(2);
k = size(b);
k = k(2);

cvx_begin quiet
    variable x(n,k)
    minimize (0.5*square_pos(norm(A*x-b,'fro'))+mu*sum(norms(x,2,2)))
cvx_end
out.fval = cvx_optval;
iter = 12;
end