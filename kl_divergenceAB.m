function kl_div = kl_divergenceAB(P, Q)
    % Compute the Kullback-Leibler Divergence between two discrete probability distributions.

    % Ensure that P and Q are probability distributions (sum to 1)
    P = P ./ sum(P);
    Q = Q ./ sum(Q);

    % Compute KL divergence
    kl_div = sum(P .* log(P ./ Q), 'omitnan');
end