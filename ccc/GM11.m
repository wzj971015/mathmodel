x0 =[83 95 130 141 156 185];
L = length(x0);
x1 = zeros(1,L);
x1(1) = x0(1);
y = x0(2:end);
for k = 2:L
    x1(k) = x1(k-1) + x0(k);
end
B = ones(L-1,2);
for k = 1:(L-1)
    B(k,1) = -1/2*(x1(k+1)+x1(k));
end
U = inv((B'*B))*B'*y';
x_predict1 = zeros(1,L);
x_predict1(1) = x1(1);
for k = 2:L
    x_predict1(k) = (x1(1)-U(2)/U(1))*exp(-U(1)*(k-1))+U(2)/U(1);
end
x_predict0 = x_predict1;
for k = 2:L
    x_predict0(k) = x_predict1(k)-x_predict1(k-1);
end
res = x0 - x_predict0;
raltive_res = res./x0;
res_sigma = std(res(2:end));
x0_sigma = std(x0);
x0_mean = mean(x0);
C = res_sigma/x0_sigma;
idx = find(abs(res-repmat(x0_mean,1,L)<0.6745*x0_sigma));
P = length(idx)/L;
