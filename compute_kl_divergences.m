function kl_divergences = compute_kl_divergences(data)
    % Initialize matrix to store KL divergences
    n = size(data, 1);
    kl_divergences = zeros(n, n);

    % Loop through pairs of distributions and compute KL Divergence
    for i = 1:n
        for j = 1:n
            kl_divergences(i, j) = kl_divergenceAB(data(i, :), data(j, :));
        end
    end
end
