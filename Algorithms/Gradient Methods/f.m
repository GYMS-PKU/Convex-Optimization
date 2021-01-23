function [y] = f(A,b,x,mu)
y = 0.5*(norm(A*x-b,'fro')^2)+mu*sum(norms(x,2,2));
end

