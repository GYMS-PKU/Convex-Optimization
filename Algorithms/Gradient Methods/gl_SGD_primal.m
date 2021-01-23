function [x,iter,out,allval] = gl_SGD_primal(x0,A,b,mu0,op) %涉及到特征值的变步长
[~,p] = size(op);

if p >= 1 %默认参数
    maxiter = op(1);
else
    maxiter =280;
    pp = 10; %默认连续化策略参数
    most = 3.5;
end
if p >= 2 %默认参数
    pp = op(2);
end
if p >= 3 %默认参数
    most = op(3);
end

mu = mu0*(pp^(4)); 
[x,iter,out] = gl_sgd1(x0,A,b,mu,[15,mu0]);
mu = mu/pp;
for i = 1:3
    [x,iter1,out1] = gl_sgd1(x,A,b,mu,[maxiter,mu0]);
    out=[out out1];
    iter = iter+iter1;
    mu = mu/pp;
end
[x,iter1,out1] = gl_sgd1(x,A,b,5*mu0,[maxiter/2,mu0]);
out=[out out1];
iter = iter+iter1;
[x,iter1,out1] = gl_sgd1(x,A,b,2*mu0,[maxiter/2,mu0]);
out=[out out1];
iter = iter+iter1;
[x,iter1,out1] = gl_sgd1(x,A,b,mu0,[most*maxiter,mu0]);
allval=[out out1];
out2.fval = allval(end);
out2.all = allval;
out = out2;
iter = iter+iter1;
end

function [x,iter,out] = gl_sgd1(x0,A,b,mu,op) %涉及到特征值的变步长
[~,n] = size(A);
[~,L] = size(b);
xk = x0;
val =[]; 
xs = []; 
maxiter = op(1);
mu0 = op(2);
alpha = 1/(max(svd(A)))^2;
fk = 0;
fkk = 0;
count = 1;
        while count <= maxiter
            if fk == 0
                fk = f(A,b,xk,0.01);
            end
            gk = A'*(A*xk-b)+mu*get_subgradient(xk,n,L);
            if mu == 0.01
                xkk= xk-alpha*100/(max([count 100]))*gk;
            else
                xkk= xk-alpha*gk;
                fkk = f(A,b,xkk,0.01);
            end
            val = [val fk];
            xs = [xs xk];
            if mu > mu0
                if count > 70 && abs(fkk-val(end-70))/val(end-70)<10^(-3)
                    break
                end
            end
            if mu <= mu0+(1e-12)
                i = -12;
            else
                i = -7;
            end
            if abs(fk-fkk)/fk < 10^(-12)
                break
            end
            xk = xkk;
            fk = fkk;
            count = count+1;
        end
 
p = find(val == min(val));
x = xs(:,(p-1)*L+1:p*L);
iter = count-1;
out = [val,val(p)];
end

function [x1] = get_subgradient(x,n,L)
x1 = zeros(n,L);
for i = 1:n 
    len = norm(x(i,:));
    if len > 0
    for j = 1:L
        x1(i,j) = x(i,j)/len;
    end
    end
end
end


