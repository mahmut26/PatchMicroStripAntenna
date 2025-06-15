clc; clear;

num_samples = 300;     
num_variables = 3;     


lhs_samples = lhsdesign(num_samples, num_variables, 'criterion', 'maximin', 'iterations', 100);


var_min = [7, 10, 0.5];
var_max = [80, 95, 3.2];

scaled_samples = lhs_samples .* (var_max - var_min) + var_min;

R1 = scaled_samples(:,1);
R2 = scaled_samples(:,2);
R3 = scaled_samples(:,3);

extra_lhs = lhsdesign(num_samples, 2, 'criterion', 'maximin', 'iterations', 100);

R4 = R1 + (12 + 20 * extra_lhs(:,1));   
R5 = R2 + (12 + 20 * extra_lhs(:,2));   

scaled_samples_lhs = [R1, R2, R3, R4, R5];

rng(1);
indices = randperm(num_samples);
scaled_samples_lhs = scaled_samples_lhs(indices, :);  


train_samples = scaled_samples_lhs(1:50, :);
test_samples  = scaled_samples_lhs(51:100, :);

save_batchess(train_samples, "train");
save_batchess(test_samples,  "test");

function save_batchess(data, prefix)
    batch_size = 10;
    num_batches = size(data, 1) / batch_size;

    for i = 1:num_batches
        idx_start = (i-1)*batch_size + 1;
        idx_end   = i*batch_size;
        batch = data(idx_start:idx_end, :);

        filename = sprintf("%s_batch_%d.mat", prefix, i);
        save(filename, 'batch');
    end
end
