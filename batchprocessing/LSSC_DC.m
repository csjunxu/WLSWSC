% single weighted: weighted least square and sparse coding framework
function  X = LSSC_DC(Y, Wls, par)
% initialize D
[U, ~, V] = svd(Y * Y', 'econ');
D = V * U';
f_curr = 0;
for i=1:par.WWIter
    f_prev = f_curr;
    % update C by soft thresholding
    B = D' * Y;
    C = sign(B) .* max(abs(B) - Wls, 0);
    % update D
    [U, ~, V] = svd( C * Y', 'econ');
    D = V * U';
    % energy function
    DT = bsxfun(@times, Y - D * C, Wls);
    DT = norm(DT, 'fro');
    %     DT = DT(:)'*DT(:);
    RT = sum(sum(abs(C)));
    f_curr = 0.5 * DT ^ 2 + par.lambdasc * RT;
%     fprintf('WLSSC Energy, %d th: %2.8f\n', i, f_curr);
    if (abs(f_prev - f_curr) / f_curr < par.epsilon)
        break;
    end
end
% update X
X = D * C;
return;
