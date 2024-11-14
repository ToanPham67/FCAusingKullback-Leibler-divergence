function kl_divergence = compute_kl_divergence(P, Q)
    % Ensure that P and Q are valid probability distributions
%     if abs(sum(P) - 1) > eps || abs(sum(Q) - 1) > eps
%         error('Input vectors must represent valid probability distributions.');
%     end

    % Compute KL Divergence
    kl_divergence = sum(P .* log(P ./ Q), 'omitnan');
end