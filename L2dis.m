function L2dis = L2dis(data)
    % Initialize matrix to store KL divergences
    n = size(data, 1);
    L2dis = zeros(n, n);

    % Loop through pairs of distributions and compute KL Divergence
    for i = 1:n
        for j = 1:n
            L2dis(i, j) = l2(data(i, :), data(j, :));
        end
    end
end
