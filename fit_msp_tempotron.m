% FIT_MSP_TEMPOTRON(ts, trials, labels, w, V_thresh, V_rest, tau_m, tau_s, lr, n_iter, optimizer, fn_target)
%  train multi-spike tempotron on given trials and labels
%   ts: time vector
%   trials: cell array of trials. Each entry is a cell array of input spike times
%   labels: labels (cumulative number of output spikes) for each trial
%   w: synaptic efficiencies / weights
%   V_thresh: spiking threshold of neuron model (see MSPTempotron)
%   V_rest: resting potential of neuron model (see MSPTempotron)
%   tau_m: membrane time constant of neuron model (see MSPTempotron)
%   tau_s: synapse time constant of neuron model (see MSPTempotron)
%   lr: learning rate parameter
%   n_iter: total number of iterations performed
%   optimizer: one of 'sgd', 'adagrad', 'rmsprop', 'adam'
%   fn_target: function handle to custom error function with signature fn(sample_idx, t_out, target_cum_reward)

function [w, t_crit, dv_dw, errs, outputs, w_hist, anneal_lr, t_adam] = fit_msp_tempotron(ts, trials, labels, w, V_thresh, V_rest, tau_m, tau_s, lr, n_iter, optimizer, fn_target, optimizer_meta_param)
    
    if nargin < 12
        fn_target = [];
    end
    
    if nargin < 11
       optimizer = 'rmsprop'; 
    end
    
    if nargin < 13
        optimizer_meta_param = 0.99;
    else
        disp(sprintf('Using %s meta_param: %.3f @ lr: %.4f', optimizer, optimizer_meta_param, lr));
    end
    
    
    dataFormatType = iscell(trials{1});
    if dataFormatType == 0
        % this means, data is formated as cell array with spikes times as
        % columns (per synapse)
        N_syn = size(trials(1,:), 2);
    else
        N_syn = length(trials{1});
    end
    
    errs = [];
    outputs = zeros(1, size(trials, 1));
    d_momentum = zeros(1, N_syn);
    t_crit = 0;
    dv_dw = [];
    w_hist = [];
    grad_cache = zeros(1, N_syn); %adagrad / RMSprop gradient cache
    eps = 10^-6;
    momentum_mu = optimizer_meta_param;    % momentum hyper param
    rms_decay_rate = optimizer_meta_param; % RMSprop leak
    lr_step = 100;
    lr_step_size = 0.001; % annealing step size
    lr_min = 0.0001;
    anneal_lr = lr;
    
    % ADAM hyper params
    beta1 = 0.9;
    beta2 = 0.999;
    m = grad_cache;
    v = grad_cache;
    t_adam = max(1, n_iter);
    
    shuffle_idx = randperm(size(trials, 1));
    profile_start = tic;
    for i=1:size(trials,1)
        % determine format of pattern
        if dataFormatType == 0
            pattern = cell(trials(i,:));
        else
            pattern = trials{i};
        end
        
        target = labels(i);
        
        if mod(i, 10) == 0
           tElapsed = toc(profile_start);
           %disp(sprintf('   trial %d [%.3f sec]', i, tElapsed)); 
           profile_start = tic;
        end
        
        [v_t, t_out, t_out_idx, v_unreset, ~, ~, V_0, tau_m, tau_s] = MSPTempotron(ts, pattern, w, V_thresh, V_rest, tau_m, tau_s);
        outputs(i) = length(t_out);
        % keep track on errors
        if (~isempty(fn_target))
            %err = fn_target(shuffle_idx(i), t_out, labels(shuffle_idx(i))) - outputs(shuffle_idx(i));
            err = fn_target(i, t_out, target);
            %disp(sprintf('   err=%d out=%d target=%d', err, outputs(shuffle_idx(i)), labels(shuffle_idx(i))));
        else
            err = target - length(t_out);
        end
        
        if (any(isnan(v_t)))
           error('NaNs !!!'); 
        end
        
        if (mod(t_adam, lr_step) == 0)
           anneal_lr = max(anneal_lr - lr_step_size, lr_min); 
        end
        
        if (err ~= 0) % perform training only on error trial
             %disp(sprintf('  trial %d | %d -> %d | %d | %.2f | %.2f ', i, outputs(i), labels(i), errs(i), norm(w), mean(v_t)));
             
            t_adam = t_adam + 1;
            errs = [errs err];
            
            [pks, pks_idx, t_crit, d_w, dw_dir, dv_dw] = msp_grad(V_0, V_thresh, pattern, w, ts, v_t, v_unreset, t_out, t_out_idx, err, tau_m, tau_s);
                        
            if all(d_w==0)
               disp(sprintf('WARN: d_w is zero !')); 
            end
            
            if strcmpi(optimizer, 'adagrad') == 1
                % ADAgrad optimizer
                %disp('** adagrad');
                grad_cache = grad_cache + d_w.^2;
                delta = (((dw_dir * lr) .* d_w) ./ (sqrt(grad_cache) + eps));
            elseif strcmpi(optimizer, 'rmsprop') == 1
                % RMSprop
                %disp('** RMSprop');
                grad_cache = rms_decay_rate .* grad_cache + (1 - rms_decay_rate) .* d_w.^2;
                delta = (((dw_dir * lr) .* d_w) ./ (sqrt(grad_cache) + eps));
                  
            elseif strcmpi(optimizer, 'adam') == 1
                % ADAM
                %disp('** ADAM');
                m = beta1 .* m + (1-beta1) .* (dw_dir .* d_w);
                mt = m ./ (1-beta1.^t_adam);
                v = beta2 .* v + (1-beta2) .* (d_w.^2);
                vt = v ./ (1-beta2.^t_adam);
                delta = (((dw_dir * lr) .* mt) ./ (sqrt(vt) + eps));
            elseif strcmpi(optimizer, 'nesterov') == 1
                %disp('** Nesterov');
                % Nesterov Momentum
                error("nesterov momentum not yet implemented - please use vanilla momentum for now.");
                %d = (lr .* d_w) + (momentum_mu .* d);
                %delta = (dw_dir .* d); 
            elseif strcmpi(optimizer, 'momentum') == 1
                %disp('** Momentum');
                % Momentum
                d_momentum = ((dw_dir * lr) .* d_w) + (momentum_mu .* d_momentum);
                delta = d_momentum; 
            
            else
                %default: vanilla SGD
                %disp('** SGD');
                delta = ((dw_dir * lr) .* d_w); % regular gradient-based learning
            end
            
            % update weights
            w = w + delta;
            w_hist = [w_hist; w];
        end
    end
end