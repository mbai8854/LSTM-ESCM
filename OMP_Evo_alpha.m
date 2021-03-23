function [C,U] = OMP_Evo_alpha(X, K, thr,alpha2,C_pre,U_pre)
% set U_pre = 0 for using fixed alfa
if (nargin < 4)
    alpha2 = 0;
end

if (nargin < 5)
    C_pre = 0;
end

if (nargin < 6)
    U_pre = 0;
end

[~, N] = size(X);
MEMORY_TOTAL = 0.1 * 10^20; % memory available for double precision.

Xn = X; % assume that data is column normalized. If not, uncomment the following.
% Xn = cnormalize(X);

% set U_pre = 0 for using fixed alfa
alpha = alpha2;
if size(U_pre,2) > 1
    f = @(u) norm(Xn-Xn*(u*U_pre+C_pre*(u-1)),'fro')^2;
    alpha = (alpha+gss(f,0,1))/2;
end

S = ones(N, K); % Support set
Ind = repmat((1:N)', 1, K);
Val = zeros(N, K); % Nonzero Value
t_vec = ones(N, 1) * K;

if alpha ~= 0
    res = (Xn-(1-alpha)*Xn*C_pre)/alpha;
else
    res = Xn; % residual
end

for t = 1:K
    blockSize = round(MEMORY_TOTAL / N);
    counter = 0;
    while(1)
        mask = counter+1 : min(counter + blockSize, N);
        I = abs(X' * res(:, mask));
        I(counter+1:N+1:end) = 0; % set diagonal = 0
        [~, J] = max(I, [], 1);
        S(mask, t) = J;
        counter = counter + blockSize;
        if counter >= N
            break;
        end
    end
    
    if t ~= K % not the last step. compute residual
        for iN = 1:N
            if t_vec(iN) == K % termination has not been reached
                B = Xn(:, S(iN, 1:t));
                res(:, iN) = Xn(:, iN) - B* (B \ Xn(:, iN));
                if sum( res(:, iN).^2 ) < thr
                    t_vec(iN) = t;
                end
            end
        end
    end
    if sum(t_vec == K) == 0
        break;
    end
    %     fprintf('Step %d in %d\n', t, K);
end

% compute coefficients
for iN = 1:N
    Val(iN, 1:t_vec(iN)) = (X(:, S(iN, 1:t_vec(iN))) \ X(:, iN))'; % use X rather than Xn
end

C = sparse( S(:), Ind(:), Val(:), N, N );
U = C;


if alpha ~= 0
    C = alpha*C+(1-alpha)*C_pre;
end