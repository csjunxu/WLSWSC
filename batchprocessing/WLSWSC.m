% double weighted: weighted least square and weighted sparse coding framework
function  X = WLSWSC(Y, Wls, par)
% initialize D and S
[D, S, ~] = svd(full(Y), 'econ');
S = diag(S);
% f_curr = 0;
C = zeros(size(D, 1), size(Y, 2));
for i=1:par.WWIter
    %     f_prev = f_curr;
    C_prev = C;
    % update W for weighted sparse coding
    Wsc = bsxfun(@rdivide, par.lambdasc * Wls .^ 2, sqrt(S) + eps );
    % update C by soft thresholding
    B = D' * Y;
    C = sign(B) .* max(abs(B) - Wsc, 0);
    % update D and S
    if par.model == 1
        % model 1
        CW = bsxfun(@times, C, Wls);
        [U, S, V] = svd( CW * Y', 'econ');
        %         [U, S, V] = svd(full(Y), 'econ');
    else
        % model 2
        CW = bsxfun(@times, C, Wls);
        YW = bsxfun(@times, Y, Wls);
        %         [U, S, V] = svd(full(Y), 'econ');
        [U, S, V] = svd( CW * YW', 'econ');
    end
    D = U * V';
    %     D = U;
    S = diag(S);
    
    residual = norm(C - C_prev, 1);
    if residual < par.epsilon
        %     if (abs(f_prev - f_curr) / f_curr < par.epsilon)
        break;
    end
%     fprintf('Residual of C: %2.4f\n', residual);
    %     DT = bsxfun(@times, Y - D * C, Wls);
    %     DT = DT(:)'*DT(:) / 2;
    %     RT = Wsc .*  C;
    %     RT = norm(RT, 1);
    %     f_curr = DT + RT;
    %     if (abs(f_prev - f_curr) / f_curr < par.epsilon)
    %         break;
    %     end
    %     fprintf('WLSWSC Energy: %2.8f\n', f_curr);
end
% update X
X = D * C;
return;
