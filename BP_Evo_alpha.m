function [C2,U] = BP_Evo_alpha(Y,K,affine,alpha,thr,maxIter,alfa2,C_pre,U_pre)
% set U_pre = 0 for using fixed alfa
if (nargin < 3)
    % default subspaces are linear
    affine = false; 
end
if (nargin < 4)
    % default regularizarion parameters
    alpha = 800;
end
if (nargin < 5)
    % default coefficient error threshold to stop ADMM
    % default linear system error threshold to stop ADMM
    thr = 2*10^-4; 
end
if (nargin < 6)
    % default maximum number of iterations of ADMM
    maxIter = 200; 
end

if (nargin < 7)
    alfa2 = 0; 
end

if (nargin < 8)
    C_pre = 0; 
end

if (nargin < 9)
    U_pre = 0; 
end

if (length(alpha) == 1)
    alpha1 = alpha(1);
    alpha2 = alpha(1);
elseif (length(alpha) == 2)
    alpha1 = alpha(1);
    alpha2 = alpha(2);
end

if (length(thr) == 1)
    thr1 = thr(1);
    thr2 = thr(1);
elseif (length(thr) == 2)
    thr1 = thr(1);
    thr2 = thr(2);
end
% set U_pre = 0 for using fixed alfa
alfa = alfa2;
if size(U_pre,2) > 1
    f = @(u) norm(Y-Y*(u*U_pre+C_pre*(u-1)),'fro')^2;
    alfa = (alfa+gss(f,0,1))/2;
end

N = size(Y,2);
Ybar = (1-alfa)*Y+alfa*Y*C_pre;
% setting penalty parameters for the ADMM
mu1 = alpha1 * 1/computeLambda_mat(Y);  % I added 2 to make it denser
mu2 = alpha2 * 1;



if (~affine)
    % initialization
    A = inv(mu1*(Y'*Y)+mu2*eye(N));
    C1 = zeros(N,N);
    Lambda2 = zeros(N,N);
    err1 = 10*thr1; err2 = 10*thr2;
    i = 1;
    % ADMM iterations
    while ( err1(i) > thr1 && i < maxIter )
        % updating Z
        Z = A * (mu1*(Y'*Ybar)+mu2*(C1-Lambda2/mu2));
        Z = Z - diag(diag(Z));
        % updating C
        C2 = max(0,(abs(Z+Lambda2/mu2) - 1/mu2*ones(N))) .* sign(Z+Lambda2/mu2);
        C2 = C2 - diag(diag(C2));
        % updating Lagrange multipliers
        Lambda2 = Lambda2 + mu2 * (Z - C2);
        % computing errors
        err1(i+1) = errorCoef(Z,C2);
        err2(i+1) = errorLinSys(Y,Z);
        %
        C1 = C2;
        i = i + 1;
    end
    % fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end),err2(end),i);
else
    % initialization
    A = inv(mu1*(Y'*Y)+mu2*eye(N)+mu2*ones(N,N)); % we cache this, better ways to do this? see ADMM paper. Matrix invdersion lemma?
    C1 = zeros(N,N);
    Lambda2 = zeros(N,N); % Lambda2 is Delta in the paper
    lambda3 = zeros(1,N); % lambda3 is delta transpose in the paper
    err1 = 10*thr1; err2 = 10*thr2; err3 = 10*thr1;
    i = 1;
    % ADMM iterations
    while ( (err1(i) > thr1 || err3(i) > thr1) && i < maxIter )
        % updating Z
        Z = A * (mu1*(Y'*Ybar)+mu2*(C1-Lambda2/mu2)+mu2*ones(N,1)*(ones(1,N)-lambda3/mu2)); % This Z is A in the paper
        Z = Z - diag(diag(Z)); % This Z is J in the paper
        % updating C
        C2 = max(0,(abs(Z+Lambda2/mu2) - 1/mu2*ones(N))) .* sign(Z+Lambda2/mu2); % soft thresholding, C2 is C in the paper
        C2 = C2 - diag(diag(C2));
        % updating Lagrange multipliers
        Lambda2 = Lambda2 + mu2 * (Z - C2);
        lambda3 = lambda3 + mu2 * (ones(1,N)*Z - ones(1,N));
        % computing errors
        err1(i+1) = errorCoef(Z,C2);
        err2(i+1) = errorLinSys(Y,Z);
        err3(i+1) = errorCoef(ones(1,N)*Z,ones(1,N));
        %
        C1 = C2;
        i = i + 1;
    end
    % fprintf('err1: %2.4f, err2: %2.4f, err3: %2.4f, iter: %3.0f \n',err1(end),err2(end),err3(end),i);
end

U = C2;
if alfa ~= 0
    C2 = alpha*C2+(1-alpha)*C_pre;
end