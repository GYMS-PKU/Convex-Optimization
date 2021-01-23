function [x, iter, out] = gl_gdbb(x0, A, b, mu0, op)
[~,p] = size(op);
iter = 0;
val = [];
if p < 2
    mu = mu0*500000;
    eta = 0.85;
else
    mu = op(1);
    eta = op(2);
end
    count = 1;
    while mu > mu0
        es1 = 10^-(1+count);
        es2 = 10^-(4+count);
        sigma = 10^(-3)*mu;
        [x0,iter1,out] =  sm(x0, A, b, mu, [sigma,es1,es2,eta,3000]);
        val = [val,out.all];
        iter = iter+iter1;
        count = count+1;
        mu = max(mu/10,mu0);
    end
    es1 = 10^-(count+1);
    es2 = 0;
    sigma = 10^(-4)*mu;
    [x,iter1,out] =  sm(x0, A, b, mu0, [sigma,es1,es2,eta,400]);
    iter = iter1 + iter;
    val = [val,out.all];
    out1.fval = out.fval;
    out1.all = val;
    out = out1;
end

function [xk, iter, out] = sm(x0, A, b, mu, op) %Hongchao and Hagger’s method
[~,n] = size(A);
[~,L] = size(b);
val = [];
k = 0;
xk = x0;
sigma = op(1); 
Ck = fs(xk,A,b,mu,sigma); 
Qk = 1; 
es1 = op(2); 
es2 = op(3); 
c1 = 1E-3; 
t = 0.1;
eta = op(4); 
maxiter = op(5);
alpha_min = 1e-12; 
alpha_max = 1e12; 
alpha = 2e-4;
gk = A'*(A*xk-b)+mu*smooth(xk,n,L,sigma); 
normgk = sum(sum(gk.*gk))^0.5;
y = Ck;
while (normgk > es1) && (k <maxiter)
    backCnt=0;
    xkk = xk-alpha*gk;
    while (fs(xkk,A,b,mu,sigma) > Ck-c1*alpha*normgk^2) && (backCnt<10)
        alpha = alpha*t;
        xkk = xk-alpha*gk;
        backCnt=backCnt+1;
    end
    ds = -alpha * gk;
    gkk = A'*(A*xkk-b)+mu*smooth(xkk,n,L,sigma);
    dg = gkk - gk;
    gk = gkk;
    normgk = norm(gk,'fro');
    fkk = fs(xkk,A,b,mu,sigma);
    val = [val,fkk];
    Qkk = eta*Qk +1;
    Ck = (eta*Qk*Ck+fkk)/Qkk;
    Qk = Qkk;
    normDs = sum(sum(ds.*ds));
    alpha = max(min(normDs /(sum(sum(ds.*dg))),alpha_max),alpha_min);
    k = k+1;
    xk = xkk;
    if abs(fkk-y)<=es2
        break
    else
        y = fkk;
    end
end
iter = k;
out.all = val;
out.fval = 0.5*norm(A*xk-b,'fro')^2+mu*sum(norms(xk,2,2));
end

function y = fs(x,A,b,mu,sigma) %光滑化函数的值
   k = 0;
   [n,~] = size(x);
   for i = 1:n
       t = norm(x(i,:),2);
       if (t <= sigma)
           k = k+t^2/(2*sigma);
       else
           k = k+t-sigma/2;
       end 
   end
   y = 0.5*(norm(A*x-b,'fro'))^2+mu*k;
end

function [x1] = smooth(x,n,L,sigma) %返回光滑化的梯度
x1 = zeros(n,L); 
for i = 1:n 
    len = norm(x(i,:));
    if len > sigma
        for j = 1:L
            x1(i,j) = x(i,j)/len;
        end
    else
        for j = 1:L
            x1(i,j) = x(i,j)/sigma;
        end
    end
end
end