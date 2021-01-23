% function Test_group_lasso

% min 0.5 * ||A * x - b||_2^2 + mu * ||x||_{1,2}

% generate data
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);

n = 512;
m = 256;
A = randn(m,n); %
k = round(n*0.1); l = 2;
A = randn(m,n);
p = randperm(n); p = p(1:k);
u = zeros(n,l);  u(p,:) = randn(k,l);  
b = A*u;
mu = 1e-2;

x0 = randn(n, l);

errfun = @(x1, x2) norm(x1 - x2, 'fro') / (1 + norm(x1,'fro'));
errfun_exact = @(x) norm(x - u, 'fro') / (1 + norm(u,'fro'));
sparisity = @(x) sum(abs(x(:)) > 1E-6 * max(abs(x(:)))) /(n*l);

% cvx calling mosek
opts1 = []; % modify options
tic;
[x1, iter1, out1] = gl_cvx_mosek(x0, A, b, mu, opts1);
t1 = toc;

% cvx calling gurobi
opts2 = []; % modify options
tic;
[x2, iter2, out2] = gl_cvx_gurobi(x0, A, b, mu, opts2);
t2 = toc;

% Subgradient Method
opts5 = []; % modify options
tic;
[x5, iter5, out5] = gl_SGD_primal(x0, A, b, mu, opts5);
t5 = toc;

% Gradient Method for the Smoothed Primal Problem
opts6 = []; % modify option
tic;
[x6, iter6, out6] = gl_GD_primal(x0, A, b, mu, opts6);
t6 = toc;

% Gradient Method for the Smoothed Primal Problem With BB method
opts7 = []; % modify option
tic;
[x7, iter7, out7] = gl_gdbb(x0, A, b, mu, opts7);
t7 = toc;

% Print the results
fprintf('     CVX-Mosek: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t1, iter1, out1.fval, sparisity(x1), errfun_exact(x1), errfun(x1, x1), errfun(x2, x1));
fprintf('    CVX-Gurobi: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t2, iter2, out2.fval, sparisity(x2), errfun_exact(x2), errfun(x1, x2), errfun(x2, x2));

fprintf('    SGD Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t5, iter5, out5.fval, sparisity(x5), errfun_exact(x5), errfun(x1, x5), errfun(x2, x5));
fprintf('     GD Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t6, iter6, out6.fval, sparisity(x6), errfun_exact(x6), errfun(x1, x6), errfun(x2, x6));
fprintf('     GDbb Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t7, iter7, out7.fval, sparisity(x7), errfun_exact(x7), errfun(x1, x7), errfun(x2, x7));

%����Ϊ��ͼ����Ĳ���
%%
%Ĭ���������
[~,~,out] = gl_SGD_primal(x0, A, b, mu, opts5);
semilogy(out.all);
hold on;
[~,~,out] = gl_gdbb(x0, A, b, mu, opts5);
semilogy(out.all);
hold on;

xlabel('��������');
ylabel('����ֵ');
title('���ݶ��½��㷨��������');
legend('SGD','GD(BB)');
grid on;

%%
%���������
for i = 1:5
    seed = 97006855+i*1000;
    ss = RandStream('mt19937ar','Seed',seed);
    RandStream.setGlobalStream(ss);
    n = 512;
    m = 256;
    A = randn(m,n); 
    k = round(n*0.1); l = 2;
    A = randn(m,n);
    p = randperm(n); p = p(1:k);
    u = zeros(n,l);  u(p,:) = randn(k,l);  
    b = A*u;
    mu = 1e-2;
    x0 = randn(n, l);
    [~,~,~,allval] = gl_SGD_primal(x0, A, b, mu, opts5);
    loglog(allval);
    hold on;
    [~,~,out] = gl_gdbb(x0, A, b, mu, opts5);
    loglog(out.all);
    hold on;
    [~,~,~,allval] = gl_GD_primal(x0, A, b, mu, opts5);
    loglog(allval);
    hold on;
end
xlabel('��������');
ylabel('����ֵ');
title('�㷨��������');
legend('SGD','GD','GDBB');
grid on;

%%
val = [];
x = [];
spar = [];
for i = 1:14
    [x1,~,~,allval] = gl_GD_primal(x0, A, b, mu, [250,10,6.5,10^(-i)]);
    loglog(allval);
    val = [val,allval(end)];
    spar = [spar,sparisity(x1)];
    x = [x,i];
    hold on;
end
xlabel('��������');
ylabel('����ֵ');
title('��ͬ\sigmaѡȡ�µ���������');
grid on;
%%
plotyy(x(2:end),val(2:end),x(2:end),spar(2:end));
xlabel('-log\sigma');
title('��ͬ\sigmaѡȡ�Խ��Ӱ��');
legend('����ֵ','ϡ���');
grid on;

%%
%���������20��ʵ��
SGD = [];
GD = [];
BB = [];
cvxmosek = [];
cvxgurobi = [];
for i = 1:20
    seed = 97006855+i*1000;
    ss = RandStream('mt19937ar','Seed',seed);
    RandStream.setGlobalStream(ss);
    n = 512;
    m = 256;
    A = randn(m,n); 
    k = round(n*0.1); l = 2;
    A = randn(m,n);
    p = randperm(n); p = p(1:k);
    u = zeros(n,l);  u(p,:) = randn(k,l);  
    b = A*u;
    mu = 1e-2;
    x0 = randn(n, l);
    errfun_exact = @(x) norm(x - u, 'fro') / (1 + norm(u,'fro'));
    tic;
    [x,iter,out] = gl_cvx_mosek(x0, A, b, mu, opts5);
    t = toc;
    cvxmosek = [cvxmosek,[sparisity(x);iter;out.fval;t;errfun_exact(x)]];
    tic;
    [x,iter,out] = gl_cvx_gurobi(x0, A, b, mu, opts5);
    t = toc;
    cvxgurobi = [cvxgurobi,[sparisity(x);iter;out.fval;t;errfun_exact(x)]];
    tic;
    [x,iter,out] = gl_SGD_primal(x0, A, b, mu, opts5);
    t = toc;
    SGD = [SGD,[sparisity(x);iter;out.fval;t;errfun_exact(x)]];
    tic;
    [x,iter,out] = gl_GD_primal(x0, A, b, mu, opts5);
    t = toc;
    GD = [GD,[sparisity(x);iter;out.fval;t;errfun_exact(x)]];
    tic;
    [x, iter, out] = gl_gdbb(x0, A, b, mu, opts7);
    t = toc;
    BB = [BB,[sparisity(x);iter;out.fval;t;errfun_exact(x)]];
end
%%
i = 1;
plot(cvxmosek(i,:));
hold on;
plot(cvxgurobi(i,:));
hold on;
plot(SGD(i,:));
hold on;
plot(GD(i,:));
hold on;
plot(BB(i,:));
hold on;
xlabel('ʵ�����');
ylabel('ϡ���');
title('��ͬ��������ϡ��ȱȽ�')
legend('cvxmosek','cvxgurobi','SGD','GD','BB');
grid on;
%%
i = 2;
plot(SGD(i,:));
hold on;
plot(GD(i,:));
hold on;
plot(BB(i,:));
hold on;
xlabel('ʵ�����');
ylabel('��������');
title('��ͬ�������ĵ��������Ƚ�')
legend('SGD','GD','BB');
grid on;
%%
i = 3;
plot(cvxmosek(i,:)-cvxmosek(i,:));
hold on;
plot(cvxgurobi(i,:)-cvxmosek(i,:));
hold on;
plot(SGD(i,:)-cvxmosek(i,:));
hold on;
plot(GD(i,:)-cvxmosek(i,:));
hold on;
plot(BB(i,:)-cvxmosek(i,:));
hold on;
xlabel('ʵ�����');
ylabel('����ֵ');
title('��ͬ������������ֵ��cvxmosek����ֵ�Ƚ�')
legend('cvxmosek','cvxgurobi','SGD','GD','BB');
grid on;
%%
i = 4;
plot(cvxmosek(i,:));
hold on;
plot(cvxgurobi(i,:));
hold on;
plot(SGD(i,:));
hold on;
plot(GD(i,:));
hold on;
plot(BB(i,:));
hold on;
xlabel('ʵ�����');
ylabel('��ʱ');
title('��ͬ����������ʱ�Ƚ�')
legend('cvxmosek','cvxgurobi','SGD','GD','BB');
grid on;

%%
mean(cvxgurobi')



