function [fc,cc,kc,alpha_c,Rc,Tc,rmse,P] = calibrateProjectorDLT(x,X)
% CALIBRATEPROJECTORDLT Estimates a pinhole projector from 3D-2D pairs.

if size(x,1) ~= 2 || size(X,1) ~= 3 || size(x,2) ~= size(X,2)
   error('Expected matching 2xN image points and 3xN world points.');
end
if size(x,2) < 6
   error('At least six 3D-2D correspondences are required.');
end

[xn,T2] = normalize2D(x);
[Xn,T3] = normalize3D(X);
n = size(x,2);
A = zeros(2*n,12);
for i = 1:n
   Xi = [Xn(:,i);1]';
   u = xn(1,i);
   v = xn(2,i);
   A(2*i-1,:) = [Xi zeros(1,4) -u*Xi];
   A(2*i,:)   = [zeros(1,4) Xi -v*Xi];
end

[~,~,V] = svd(A,0);
Pn = reshape(V(:,end),4,3)';
P = T2\Pn*T3;

[K,Rc] = rq3(P(:,1:3));
sgn = sign(diag(K));
sgn(sgn == 0) = 1;
D = diag(sgn);
K = K*D;
Rc = D*Rc;
if det(Rc) < 0
   K = -K;
   Rc = -Rc;
end
Tc = K\P(:,4);
K = K/K(3,3);
P = K*[Rc Tc];

proj = P*[X;ones(1,n)];
proj = proj(1:2,:)./proj(3,:);
rmse = sqrt(mean(sum((proj-x).^2,1)));

fc = [K(1,1);K(2,2)];
cc = K(1:2,3);
alpha_c = K(1,2)/K(1,1);
kc = zeros(5,1);

end

function [xn,T] = normalize2D(x)
c = mean(x,2);
d = sqrt(sum((x-c).^2,1));
s = sqrt(2)/mean(d);
T = [s 0 -s*c(1);0 s -s*c(2);0 0 1];
xh = T*[x;ones(1,size(x,2))];
xn = xh(1:2,:)./xh(3,:);
end

function [Xn,T] = normalize3D(X)
c = mean(X,2);
d = sqrt(sum((X-c).^2,1));
s = sqrt(3)/mean(d);
T = [s 0 0 -s*c(1);0 s 0 -s*c(2);0 0 s -s*c(3);0 0 0 1];
Xh = T*[X;ones(1,size(X,2))];
Xn = Xh(1:3,:)./Xh(4,:);
end

function [R,Q] = rq3(A)
[Q,R] = qr(flipud(A)');
R = flipud(R');
R = fliplr(R);
Q = Q';
Q = flipud(Q);
end
