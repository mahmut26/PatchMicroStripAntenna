clc; clear;


num_samples = 300;    
num_variables = 3;     


sobol_seq = sobolset(num_variables);
sobol_seq = scramble(sobol_seq,'MatousekAffineOwen');
sobol_samples = net(sobol_seq, num_samples);


var_min = [7, 10, 0.5];
var_max = [80, 95, 3.2];


scaled_samples = sobol_samples .* (var_max - var_min) + var_min;


R1 = scaled_samples(:,1);
R2 = scaled_samples(:,2);
R3 = scaled_samples(:,3);


sobol_seq_extra = sobolset(2);
sobol_seq_extra = scramble(sobol_seq_extra, 'MatousekAffineOwen');
extra_samples = net(sobol_seq_extra, num_samples);

R4 = R1 + (12 + 20 * extra_samples(:,1));   
R5 = R2 + (12 + 20 * extra_samples(:,2));   


scaled_samples_sobol = [R1, R2, R3, R4, R5];


rng(1);
indices = randperm(num_samples);
scaled_samples_sobol = scaled_samples_sobol(indices, :); 


train_samples = scaled_samples_sobol(1:50, :);
test_samples  = scaled_samples_sobol(51:100, :);


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
