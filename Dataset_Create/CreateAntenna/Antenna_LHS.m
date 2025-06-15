Fr_L = 1e9;     
Fr_U = 10e9;    
SMP = 901;       
Frequency = linspace(Fr_L, Fr_U, SMP);

scaled_samples = scaled_samples_lhs;
num_samples = size(scaled_samples, 1);
Table = zeros(SMP, 9, num_samples); 

for i = 1:num_samples
    try
        R1 = scaled_samples(i, 1);
        R2 = scaled_samples(i, 2);
        R3 = scaled_samples(i, 3);
        R4 = scaled_samples(i, 4);
        R5 = scaled_samples(i, 5);

        feed_x = min(R1 * 1e-3 / 3, R1 * 1e-3 * 0.4);
        feed_y = R2 * 1e-3 / 2;

        max_feed_x = 0.0025;  
        max_feed_y = 0.04;    


        feed_x = min(feed_x, max_feed_x);
        feed_y = min(feed_y, max_feed_y);


        feed_offset = [feed_x, feed_y];

        diel = dielectric('FR4');
        cond = metal('Copper');


        max_attempts = 5;
        attempt = 1;
        success = false;

        while ~success && attempt <= max_attempts
            try
                ant = patchMicrostrip( ...
                    'Length', R1 * 1e-3, ...
                    'Width', R2 * 1e-3, ...
                    'Height', R3 * 1e-3, ...
                    'GroundPlaneLength', R4 * 1e-3, ...
                    'GroundPlaneWidth', R5 * 1e-3, ...
                    'Substrate', diel, ...
                    'Conductor', cond, 'FeedOffset', feed_offset);

                mesh(ant, 'MaxEdgeLength', 0.02, 'MinEdgeLength', 0.0001);
                s = sparameters(ant, Frequency);
                k = s.Parameters;
                fr = s.Frequencies;


                success = true;

            catch ME_inner

                if contains(ME_inner.message, 'FeedOffset should be less than')
                    feed_x = feed_x * 0.8;
                    feed_y = feed_y * 0.8;
                    feed_offset = [feed_x, feed_y];
                    attempt = attempt + 1;
                else
                    rethrow(ME_inner);
                end
            end
        end

        if ~success
            error('FeedOffset Error.');
        end

        temp = zeros(SMP, 9);
        temp(:,1) = R1;
        temp(:,2) = R2;
        temp(:,3) = R3;
        temp(:,4) = R4;
        temp(:,5) = R5;
        temp(:,6) = fr;

        temp(:,7) = reshape(real(k(1,1,:)), [], 1);
        temp(:,8) = reshape(imag(k(1,1,:)), [], 1);
        temp(:,9) = reshape(10*log10(abs(k(1,1,:))), [], 1);

        Table(:,:,i) = temp;

        filename = sprintf('temp_sample_%d.txt', i);
        writematrix(temp, filename, 'Delimiter', 'tab');

        if mod(i, 10) == 0
            fprintf("Finished : %d/%d\n", i, num_samples);
            save("checkpoint.mat", "Table", "i");
        end

    catch ME
        warning("Errorr at %d : %s", i, ME.message);
        Table(:,:,i) = NaN;
    end
end