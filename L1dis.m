function L1dis = L1dis(data)
    % Initialize matrix to store KL divergences
    n = size(data, 1);
    L1dis = zeros(n, n);

    % Loop through pairs of distributions and compute KL Divergence
    for i = 1:n
        for j = 1:n
            L1dis(i, j) = l1(data(i, :), data(j, :));
        end
    end
end
