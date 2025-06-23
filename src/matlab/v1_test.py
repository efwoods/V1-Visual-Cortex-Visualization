root_dir = "/media/linux-pc/Stargate/gh/projects/NeuralNexus/New-Features/Thought-to-Image/V1-Visual-Cortex-Visualization"
data_root_dir = root_dir + "/data/crcns-pvc1/crcns-ringach-data/";
neuro_root_dir = "neurodata/ac1/";
movie_root_dir = "movie_frames/movie000_000.images/";
current_movie = "movie000_000_000.jpeg";
current_neuro_data = "ac1_u004_000.mat";


neuro_str = strcat(data_root_dir, neuro_root_dir, current_neuro_data);
load(neuro_str);


load_img = strcat(data_root_dir, movie_root_dir, current_movie);

%if ~strcmp(str1, str2)
%    disp('Strings are different');
%end

imread(load_img);

Total Number of Conditions
pepANA.no_conditions

Arrival Times for electrodes in pepANA.elec_list
pepANA.listOfResults{1}.repeat{1}.data{13}

Detected Waveforms at that time (48 samples of waveform snippets) (contains spikes and noise bc of low threshold) (1.6ms sampled at 30kHz at int8 precision)
pepANA.listOfResults{1}.repeat{1}.data{1}{2}

pepANA.listOfResults{1}.repeat{1}.data{1}{1}

% Length of each sample in ms
(size(pepANA.listOfResults{1}.repeat{1}.data{1}{2}', 2) / 30000) * 1000


pepANA.listOfResults{1}.repeat{1}.data{1}{2}'

pepANA.listOfResults{1}.repeat{1}.data{1}{1}

Electrodes
pepANA.elec_list

Examining Waveforms
q = pepANA.listOfResults{1}.repeat{1}.data{1}{2}

% size(q(1, :))
% one_waveform = q(1, 48);

size(q(:, :));

fig0 = figure("Name", "Waveform");
ax_0 = axes("Parent", fig0);
h = plot(ax_0, q);
numel(h)

% fig_multiple_spikes_electrode13 = figure("Name", "Channel 13 Electrode");
% ax_multi_spikes_channel13 = axes("Parent", fig_multiple_spikes_electrode13);
% plot(ax_multi_spikes_channel13, q);

SVD of Waveforms

function filtered_signal_idxs = analyzeWaveform(condition, electrode, verbose)
    if nargin < 3
        verbose = false;
    end    

    % Check if pepANA is in workspace
    if ~evalin('base', 'exist(''pepANA'', ''var'')')
        error('pepANA structure not found in the base workspace.');
    end

    % Get waveform from pepANA in base workspace
    pepANA = evalin('base', 'pepANA');
    
    % Extract waveforms
    waveforms = pepANA.listOfResults{condition}.repeat{1}.data{electrode}{2};

    % Ensure waveforms are in double precision
    waveforms = double(waveforms);

    % Perform SVD
    [~, ~, v] = svd(waveforms);

    % Plot first two principal components
    if verbose
        fig1 = figure('Name', 'Original Waveforms', 'Position', [100, 100, 800, 600]);
        ax1 = axes('Parent', fig1);
        plot(ax1, v(:, 1), v(:, 2), '.', 'MarkerSize', 10);
        hold(ax1, 'on');
        plot(ax1, 0, 0, 'r.', 'MarkerSize', 25);
        xlabel(ax1, '1st Principal Component');
        ylabel(ax1, '2nd Principal Component');
        title(ax1, sprintf('SVD Projection: Condition %d, Electrode %d', condition, electrode));
        grid(ax1, 'on');
    end
        

    idx = kmeans(double(waveforms'),2); %% cluster waveforms in 2 classes using k-means
    idx1 = find(idx==1); 
    idx2 = find(idx==2);

    % Determine which cluster is likely the signal
    mean1 = mean(waveforms(:, idx1), 2);
    mean2 = mean(waveforms(:, idx2), 2);

    total_amplitude1 = sum(abs(mean1));
    total_amplitude2 = sum(abs(mean2));

    if total_amplitude1 > total_amplitude2
        signal_idx = idx1;
        noise_idx = idx2;
    else
        signal_idx = idx2;
        noise_idx = idx1;
    end

    if verbose
        fig2 = figure("Name", "SVD of Waveforms and Plot of a Waveform");
        ax3_1 = subplot(1, 2, 1, "Parent", fig2);
        plot(ax3_1, v(noise_idx, 1), v(noise_idx, 2), 'b.', 'markersize',10); 
        hold on;
        plot(ax3_1, 0, 0, 'r.', 'markersize', 25); 
        plot(ax3_1, v(signal_idx, 1), v(signal_idx, 2), 'g.', 'markersize', 10);
    
        ax3_2 = subplot(1, 2, 2, "Parent", fig2);
        errorbar(ax3_2, 1:size(waveforms,1), mean(double(waveforms(:, signal_idx)'),1), std(double(waveforms(:, signal_idx)'),[],1), 'g'); 
        hold on;
        errorbar(ax3_2, 1:size(waveforms,1), mean(double(waveforms(:, noise_idx)'),1), std(double(waveforms(:, noise_idx)'),[],1), 'b');
    
        xlim([0 49])
    end

    filtered_signal_idxs = signal_idx;
    % mean_detected_signal = mean(double(waveforms(:, signal_idx)'),1);
end

data_root_dir = root_dir + "/data/crcns-pvc1/crcns-ringach-data/";
neuro_root_dir = "neurodata/ac1/";
movie_root_dir = "movie_frames/movie000_000.images/";
current_movie = "movie000_000_000.jpeg";
current_neuro_data = "ac1_u004_000.mat";

neuro_str = strcat(data_root_dir, neuro_root_dir, current_neuro_data);
load(neuro_str);

% Verifying the time between the samples varies
% time_stamp = pepANA.listOfResults{1}.repeat{1}.data{1}
% time_stamp{1}(1) - time_stamp{1}(2)
% time_stamp{1}(2) - time_stamp{1}(3)

Collecting all filtered data for machine learning purposes

% condition = 1
% electrode = 1
% signal = analyzeWaveform(condition, electrode, true);
% 
% % Plotting the detected spike waveform only
% fig0 = figure('Name', 'Detected Waveform', 'Position', [100, 100, 800, 600]);
% ax0 = axes('Parent', fig0);
% plot(ax0, signal)

function signal = extract_waveform(condition, electrode, filtered_signal_idx)

    % Check if pepANA is in workspace
    if ~evalin('base', 'exist(''pepANA'', ''var'')')
        error('pepANA structure not found in the base workspace.');
    end

    % Get waveform from pepANA in base workspace
    pepANA = evalin('base', 'pepANA');
    
    % Extract waveforms
    waveforms = pepANA.listOfResults{condition}.repeat{1}.data{electrode}{2};

    % Ensure waveforms are in double precision
    waveforms = double(waveforms);

    signal = double(waveforms(:, filtered_signal_idx));
end


function time = extract_waveform_timestamp(condition, electrode, filtered_signal_idx)

    % Check if pepANA is in workspace
    if ~evalin('base', 'exist(''pepANA'', ''var'')')
        error('pepANA structure not found in the base workspace.');
    end

    % Get waveform from pepANA in base workspace
    pepANA = evalin('base', 'pepANA');
    
    % Extract waveforms
    times = pepANA.listOfResults{condition}.repeat{1}.data{electrode}{1};

    time = times(1, filtered_signal_idx);
end

function stimulus_image = identify_stimulus_image(condition, electrode, filtered_signal_idx, latency, verbose)

    if nargin < 5
        verbose = false;
    end

    if nargin < 4
        latency = 60e-3;
        verbose = false;
    end

    % Check if pepANA is in workspace
    if ~evalin('base', 'exist(''pepANA'', ''var'')')
        error('pepANA structure not found in the base workspace.');
    end

    % Get waveform from pepANA in base workspace
    pepANA = evalin('base', 'pepANA');
    data_root_dir = "/media/linux-pc/Stargate/gh/projects/NeuralNexus/New-Features/Thought-to-Image/V1-Visual-Cortex-Visualization/data/crcns-pvc1/crcns-ringach-data/";

    movie_id = pepANA.listOfResults{condition}.values(1);
    segment_id = pepANA.listOfResults{condition}.values(2);

    if segment_id{1} > 0
        n_prefix_zeros = 3 - (floor(log10(segment_id{1})) + 1);
        if n_prefix_zeros > 1
            prefix_zeros_segment_id = sprintf('%s', repmat("0", 1, n_prefix_zeros));
        elseif n_prefix_zeros == 0
            prefix_zeros_segment_id = "";
        else
            prefix_zeros_segment_id = "0";
        end
    else
        prefix_zeros_segment_id = "00";
    end

    % Indices of the movie frame that appeared 60ms before a spike 
    % c = 1 % condition
    T = 3/90; % 3 frames (of the same image) at 90 Hz of the monitor (duration of each frame in seconds.)
    % electrode = 1
    spk = pepANA.listOfResults{condition}.repeat{1}.data{electrode}{1};
    spk = spk - latency; % frames of latency value (default is 60 ms) before a spike sample
    spk(spk < 0) = 0;
    frame_idxs = floor(spk/T);
    frame_idx = frame_idxs(filtered_signal_idx);
    % Create the string to show the image that is the stimulus to the detected spike
    if verbose
        sprintf('frame: %d', frame_idx)
    end
    if frame_idx > 0
        n_prefix_frame_zeros = 3 - (floor(log10(frame_idx)) + 1);
        if n_prefix_frame_zeros > 1
            prefix_zeros_frame_idx = sprintf('%s', repmat("0", 1, n_prefix_frame_zeros));
        elseif n_prefix_frame_zeros == 0
            prefix_zeros_frame_idx = "";
        else
            prefix_zeros_frame_idx = "0";
        end
    else 
        prefix_zeros_frame_idx = "00";
    end

    movie_image_string = "movie00" + int2str(movie_id{1}) + "_" + prefix_zeros_segment_id + int2str(segment_id{1}) + "_" + prefix_zeros_frame_idx + int2str(frame_idx) + ".jpeg";
    movie_image_dir = data_root_dir + "movie_frames/" + "movie00" + int2str(movie_id{1}) + "_" + prefix_zeros_segment_id + int2str(segment_id{1}) + ".images/";

    if verbose
        sprintf('movie_image_string: %s', movie_image_string)
    end

    image_path = strcat(movie_image_dir, movie_image_string);
    stimulus_image = imread(image_path);

    if verbose
        fig_img = figure("Name", "Stimulus Image");
        ax_img = axes("Parent", fig_img);
        imshow(stimulus_image,"Parent", ax_img)
    end
end


Creating and saving 120 batches of electrode data (1 per condition)
for condition = 1:1
    All_Filtered_Data = struct('condition_id', {}, "electrodes", {});
    filtered_signals = struct('electrode', {}, "data", {});
    signal_data = struct("waveform", {}, "signal_waveform_timestamp", {}, "stimulus_image", {}, "latency_from_image_to_signal_waveform_timestamp", {}, "filtered_signal_frame_idx", {});
    latency = 60e-3; % defaulting to 60 ms between signal response and image stimulus
    for electrode = 1:16
        filtered_signal_idxs = analyzeWaveform(condition, electrode, false);
        for idx = 1:length(filtered_signal_idxs)
            signal_data(idx).waveform = extract_waveform(condition, electrode, idx);
            signal_data(idx).signal_waveform_timestamp = extract_waveform_timestamp(condition, electrode, idx);
            signal_data(idx).stimulus_image = identify_stimulus_image(condition, electrode, idx, latency, false);
            signal_data(idx).latency_from_image_to_signal_waveform_timestamp = latency;
            signal_data(idx).filtered_signal_frame_idx = filtered_signal_idxs(idx);
        end
        filtered_signals(electrode).electrode = electrode;
        filtered_signals(electrode).data = signal_data;
    end
    All_Filtered_Data(condition).condition_id = condition;
    All_Filtered_Data(condition).electrodes = filtered_signals;

    % Assumptions:
    % - `All_Filtered_Data` exists in the workspace
    % - You're working with a specific condition and electrode
    % - You want to save everything except the raw image to a CSV
    
    % Define the condition and electrode index you're working with
    condition_idx = condition;
    electrode_idx = electrode;
    
    % Extract the list of signal data entries
    entries = All_Filtered_Data(condition_idx).electrodes(electrode_idx).data;
    
    % Preallocate cell array for the table
    n_entries = numel(entries);
    waveform_lengths = numel(entries(1).waveform);
    T = table();
    
    for i = 1:n_entries
        entry = entries(i);
    
        % Convert waveform to a row vector
        waveform_row = reshape(entry.waveform, 1, []);
    
        % Create a row for the table
        new_row = table( ...
            {waveform_row}, ...
            entry.signal_waveform_timestamp, ...
            entry.latency_from_image_to_signal_waveform_timestamp, ...
            entry.filtered_signal_frame_idx, ...
            'VariableNames', { ...
                'waveform', ...
                'signal_waveform_timestamp', ...
                'latency_from_image_to_signal_waveform_timestamp', ...
                'filtered_signal_frame_idx' ...
            } ...
        );
    
        % Append to table
        T = [T; new_row];
    end
    
    save_filename = sprintf("filtered_signal_data_condition_%d_electrode_%d.csv", condition_idx, electrode_idx)
    % Save to CSV file (waveform saved as string for now)
    
    file_write_path = "/media/linux-pc/Stargate/gh/projects/NeuralNexus/New-Features/Thought-to-Image/V1-Visual-Cortex-Visualization/data/filtered_signal_data/signal_csv/" + save_filename
    writetable(T, file_write_path)
    
    % Extracting the image
    % Set output folder
    output_folder = "/media/linux-pc/Stargate/gh/projects/NeuralNexus/New-Features/Thought-to-Image/V1-Visual-Cortex-Visualization/data/filtered_signal_data/stimulus_images/";
    
    
    % Loop through each condition
    for condition_number = condition_idx
        
        electrodes = All_Filtered_Data(condition_number).electrodes;
        
        % Loop through each electrode
        for electrode_number = 1:length(electrodes)
            
            signal_data = electrodes(electrode_number).data;
            
            % Loop through each entry in signal_data
            for i = 1:length(signal_data)
                
                % Extract image
                img = signal_data(i).stimulus_image;
                
                % Generate filename
                filename = sprintf('condition_%02d_electrode_%02d_sample_%03d.png', ...
                                    condition_number, electrode_number, i)
                
                % Full path
                filepath = fullfile(output_folder, filename);
                
                % Save image
                imwrite(img, filepath)
            end
        end
    end
    
    clear All_Filtered_Data;
end

All_Filtered_Data(1).electrodes(1).data

Indicating the noise and signal based on the first two dimensions
% waveform = pepANA.listOfResults{condition}.repeat{1}.data{electrode}{2};
% 
% [u, s, v] = svd(double(q));
% fig1 = figure('Name', 'Original Waveforms', "Position", [100, 100, 800, 600]);
% ax1 = axes("Parent", fig1);
% plot(ax1, v(:, 1), v(:, 2), '.', 'markersize', 10);
% hold on;
% plot(ax1, 0,0, 'r.', 'markersize', 25);

Classifying Waveforms with K-means clustering
% fig2 = figure("Name", "SVD of Waveforms and Plot of a Waveform");
% 
% idx = kmeans(double(q'),2); %% cluster waveforms in 2 classes using k-means
% idx1 = find(idx==1); 
% idx2 = find(idx==2);
% 
% ax3_1 = subplot(1, 2, 1, "Parent", fig2);
% plot(ax3_1, v(idx1, 1), v(idx1, 2), 'b.', 'markersize',10); 
% hold on;
% plot(ax3_1, 0, 0, 'r.', 'markersize', 25); 
% plot(ax3_1, v(idx2, 1), v(idx2, 2), 'g.', 'markersize', 10);
% 
% ax3_2 = subplot(1, 2, 2, "Parent", fig2)
% errorbar(ax3_2, 1:size(q,1), mean(double(q(:, idx2)'),1), std(double(q(:, idx2)'),[],1), 'g'); 
% hold on;
% errorbar(ax3_2, 1:size(q,1), mean(double(q(:, idx1)'),1), std(double(q(:, idx1)'),[],1), 'b');
% 
% xlim([0 49])

% set(gcf, 'Renderer', 'painters'); % or 'opengl'

% waveform = pepANA.listOfResults{1}.repeat{1}.data{13}{2}(1, :)
% fig0 = figure("Name", "Single Waveform");
% ax0 = axes("Parent", fig0);
% plot(ax0, waveform);

% plot(ax0, pepANA.listOfResults{1}.repeat{1}.data{13}{1})

Natural image sequences data:
% Indices of the movie frame that appeared 60ms before a spike 
c = 1 % condition
T = 3/90 % 3 frames (of the same image) at 90 Hz of the monitor (duration of each frame in seconds.)
spk = pepANA.listOfResults{c}.repeat{1}.data{13}{1};
spk = spk - 60e-3; % frames 60 ms before a spike sample
frame_idx = floor(spk/T);
frame_idx

frame_idx(1);
n_prefix_frame_zeros = 3 - (floor(log10(frame_idx(1))) + 1)

Getting the frames
function stimulus_image = identify_single_stimulus_image(condition, electrode, latency, verbose)

    if nargin < 4
        verbose = false;
    end

    if nargin < 3
        latency = 60e-3;
        verbose = false;
    end

    % Check if pepANA is in workspace
    if ~evalin('base', 'exist(''pepANA'', ''var'')')
        error('pepANA structure not found in the base workspace.');
    end

    % Get waveform from pepANA in base workspace
    pepANA = evalin('base', 'pepANA');
    data_root_dir = "/media/linux-pc/Stargate/gh/projects/NeuralNexus/New-Features/Thought-to-Image/V1-Visual-Cortex-Visualization/data/crcns-pvc1/crcns-ringach-data/";

    movie_id = pepANA.listOfResults{condition}.values(1);
    segment_id = pepANA.listOfResults{condition}.values(2);

    if segment_id{1} > 0
        n_prefix_zeros = 3 - (floor(log10(segment_id{1})) + 1);
        if n_prefix_zeros > 1
            prefix_zeros_segment_id = sprintf('%s', repmat("0", 1, n_prefix_zeros));
        else
            prefix_zeros_segment_id = "0";
        end
    else
        prefix_zeros_segment_id = "00";
    end

    % Indices of the movie frame that appeared 60ms before a spike 
    % c = 1 % condition
    T = 3/90; % 3 frames (of the same image) at 90 Hz of the monitor (duration of each frame in seconds.)
    % electrode = 1
    spk = pepANA.listOfResults{condition}.repeat{1}.data{electrode}{1};
    spk = spk - latency; % frames of latency value (default is 60 ms) before a spike sample
    frame_idx = floor(spk/T);

    % Create the string to show the image that is the stimulus to the detected spike
    if verbose
        sprintf('frame: %d', frame_idx(1))
    end
    if frame_idx(1) > 0
        n_prefix_frame_zeros = 3 - (floor(log10(frame_idx(1))) + 1);
        if n_prefix_frame_zeros > 1
            prefix_zeros_frame_idx = sprintf('%s', repmat("0", 1, n_prefix_frame_zeros));
        else
            prefix_zeros_frame_idx = "0";
        end
    else 
        prefix_zeros_frame_idx = "00";
    end

    movie_image_string = "movie00" + int2str(movie_id{1}) + "_" + prefix_zeros_segment_id + int2str(segment_id{1}) + "_" + prefix_zeros_frame_idx + int2str(frame_idx(1)) + ".jpeg";
    movie_image_dir = data_root_dir + "movie_frames/" + "movie00" + int2str(movie_id{1}) + "_" + prefix_zeros_segment_id + int2str(segment_id{1}) + ".images/";

    if verbose
        sprintf('movie_image_string: %s', movie_image_string)
    end

    image_path = strcat(movie_image_dir, movie_image_string);
    stimulus_image = imread(image_path);

    if verbose
        fig_img = figure("Name", "Stimulus Image");
        ax_img = axes("Parent", fig_img);
        imshow(stimulus_image,"Parent", ax_img)
    end
end

condition = 26
electrode = 2

spk = pepANA.listOfResults{condition}.repeat{1}.data{electrode}{1};
spk

% latency = 60e-3
latency = 0

stimulus_image = identify_single_stimulus_image(condition, electrode, latency, true)

