function [x,iter,out] = gl_cvx_mosek(x0,A,b,mu,varargin)
cvx_solver mosek;

m = size(A);
m = m(1);
n = size(A);
n = n(2);
k = size(b);
k = k(2);

% cvxµ÷ÓÃmosek
cvx_begin quiet
    variable x(n,k)
    minimize (0.5*square_pos(norm(A*x-b,'fro'))+mu*sum(norms(x,2,2)))
cvx_end;
out.fval = cvx_optval;
iter = 10;
end

