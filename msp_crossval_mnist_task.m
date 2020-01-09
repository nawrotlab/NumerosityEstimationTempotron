function [L, R, Y_oos, Y, Y_train, Y_test, W_all] = msp_crossval_mnist_task(modelName, dataSetFileName, n_folds, varargin)

args = inputParser;
defaultOptimizer='rmsprop';
validOptimizers = {'rmsprop', 'momentum'};
validCvMethods = {'KFold', 'LeaveOut'};
checkOptimizer = @(x) any(validatestring(x,validOptimizers));
checkCvMethod = @(x) any(validatestring(x,validCvMethods));

addParameter(args, 'n_epochs', 10, @isnumeric);
addParameter(args, 'max_class', 4, @isnumeric);
addParameter(args, 'min_class', 0, @isnumeric);
%addParameter(args, 'exclude_zero', 0, @isnumeric);
addParameter(args, 'model_cache', 1, @isnumeric);
addParameter(args, 'optimizer', defaultOptimizer, checkOptimizer);
addParameter(args, 'cv_type', 'KFold', checkCvMethod);
addParameter(args, 'rng_seed', 42, @isnumeric);
addParameter(args, 'sub_sample', 0.3, @isnumeric);
addParameter(args, 'run_folds', n_folds, @isnumeric);
addParameter(args, 'learn_rate', 0.0001, @isnumeric);
addParameter(args, 'optimizer_meta_parameter', 0.99999, @isnumeric);

args.KeepUnmatched = true;
parse(args,varargin{:});
n_epochs = args.Results.n_epochs;
cv_type = args.Results.cv_type;
optimizer = args.Results.optimizer;
max_class = args.Results.max_class;
min_class = args.Results.min_class;
%exclude_zero = args.Results.exclude_zero;
run_folds = args.Results.run_folds;
seed__ = args.Results.rng_seed;
sub_sample = args.Results.sub_sample;
lr = args.Results.learn_rate;
model_cache = args.Results.model_cache;
optimizer_meta_parameter = args.Results.optimizer_meta_parameter;

load(dataSetFileName);

% dataSet properties
N_samples = size(data.samples,1);
N_syn = size(data.samples, 2);
dt = 1/1000;
T = data.T_trial;     % trial duration
ts = 0:dt:T;
rng(seed__);

% neuron model
tau_m = 0.015;
tau_s = 0.005;
V_thresh = 1;
V_rest = 0;

% bring targets into correct shape, if required
if size(data.rewards, 1) ~= 1
   data.rewards = data.rewards'; 
end

targets = double(data.rewards);
samples = data.samples;
samples_oos = [];
targets_oos = [];

if max_class > 0
    filt_idx = find(targets > max_class);
    targets_oos = targets(filt_idx);
    samples_oos = samples(filt_idx, :);
    %assert(size(samples_oos, 1) > 1);
    
    filt_idx = find(targets <= max_class);
    targets = targets(filt_idx);
    samples = samples(filt_idx, :);
end

%if (exclude_zero > 0)
%    filt_idx = find(targets > 0);
%    targets = targets(filt_idx);
%    samples = samples(filt_idx, :);
%end

if (min_class > 0)
    filt_idx = find(targets >= min_class);
    targets = targets(filt_idx);
    samples = samples(filt_idx, :);
end


max_class = max(targets);
min_class = min(targets);


if sub_sample < 1.0
    c2 = cvpartition(targets,'HoldOut',sub_sample);
    disp(sprintf("(strat.) sub-sampling dataSet from %d -> %d samples", size(samples,1), c2.TestSize(1)));
    targets = targets(c2.test);
    samples = samples(c2.test, :);
end

N_samples = size(samples,1);
if strcmpi(cv_type, 'LeaveOut') == 1
c = cvpartition(targets, 'LeaveOut');
disp(sprintf("* starting LeaveOut cross-validation running %d of %d folds | max_class: %d", run_folds, c.NumTestSets, max_class));    
else
c = cvpartition(targets, 'KFold', n_folds);
disp(sprintf("* starting %d-fold cross-validation running %d folds | max_class: %d", n_folds, run_folds, max_class));
end

run_folds = min(run_folds,c.NumTestSets);

seeds = randi(100000, 1, run_folds);
R = zeros(run_folds, n_epochs, 4);
L = zeros(run_folds, n_epochs, 2);
Y = zeros(run_folds, n_epochs, min(c.TestSize()), 2);
W_all = zeros(run_folds, n_epochs+1, N_syn);
Y_oos = zeros(run_folds, n_epochs, length(targets_oos), 2);

for k=1:run_folds  
    outputFileName = sprintf('model_cache/msp_%s-%s.%d.mat', modelName, optimizer, k);
    
    if (model_cache == 1 && exist(outputFileName,'file') == 2)
        disp(sprintf('** skipped existing model: %s', outputFileName));
        continue; 
    end
    
    fold = k;
    seed = seeds(k);
    rng(k);
    
    X_train = samples(c.training(k),:);
    y_train = targets(c.training(k));
    
    randOrder = randperm(length(y_train));
 
    % randomize fold - apparently NOT done by MATLAB
    X_train = X_train(randOrder, :);
    y_train = y_train(randOrder);
    
    n_iter = 0;
    X_test = samples(c.test(k),:);
    y_test = targets(c.test(k));
    w_init = normrnd(0, 1 / N_syn, 1, N_syn);
    w_out = w_init;
    losses = zeros(n_epochs, 2);
    W = zeros(n_epochs+1, N_syn);
    W(1,:) = w_init;
    W_all(k,1,:) = w_init;
    Y_train = zeros(n_epochs, size(X_train,1), 2);
    Y_test = zeros(n_epochs, size(X_test,1), 2);
    Y_oo = zeros(n_epochs, length(targets_oos), 2);
    
    % fit single model per fold
    for i=1:n_epochs
        tic
        [w_out, ~, ~, train_err, pred_y, ~, ~, n_iter] = fit_msp_tempotron(ts, X_train, y_train, w_out, V_thresh, V_rest, tau_m, tau_s, lr, n_iter, optimizer, [], optimizer_meta_parameter);
        train_loss = mean(abs(pred_y-y_train));
        losses(i,1) = train_loss;
        W(i,:) = w_out;
        W_all(k,i,:) = w_out;
        Y_train(i, :, 1) = y_train;
        Y_train(i, :, 2) = pred_y;
        
        [mean_test_err, test_err, pred_test_y] = validate_msp_tempotron(ts, X_test, y_test, w_out, V_thresh, V_rest, tau_m, tau_s);
        test_loss = mean(abs(pred_test_y-y_test));
        losses(i,2) = test_loss;
        Y_test(i, :, 1) = y_test;
        Y_test(i, :, 2) = pred_test_y;
        
        Y(k,i,:,1) = y_test(1:min(c.TestSize()));
        Y(k,i,:,2) = pred_test_y(1:min(c.TestSize()));
        
        accu_train = (length(find(y_train == pred_y)) * 100)/length(y_train);
        rmse_train = sqrt(mean((y_train-pred_y).^2));
        
        accu_test = (length(find(y_test == pred_test_y)) * 100)/length(y_test);
        rmse_test = sqrt(mean((y_test-pred_test_y).^2));
        
        R(k,i,1) = accu_train;
        R(k,i,2) = accu_test;
        R(k,i,3) = rmse_train;
        R(k,i,4) = rmse_test;
        L(k,:,:) = losses;
        
        if size(samples_oos, 1) > 1
            [~, ~, pred_oos] = validate_msp_tempotron(ts, samples_oos, targets_oos, w_out, V_thresh, V_rest, tau_m, tau_s);
            Y_oo(i, :, 1) = targets_oos;
            Y_oo(i, :, 2) = pred_oos;
            
            Y_oos(k, i, :, 1) = targets_oos;
            Y_oos(k, i, :, 2) = pred_oos;
        end
        
        elapsed = toc;
        
        if (isempty(train_err)) % all zeros => no errors, converged
            disp(sprintf('%s learning converged after %d epochs', optimizer, i));
            break;
        end
        
        
        if (mod(i, 4) > -1)
            disp(sprintf('fold=%d [%s @ %.3f sec] | epoch=%d | lr|alpha=%.4f|%.4f | train_loss: %.3f|%.2f | test_loss: %.3f|%.2f', k, optimizer, elapsed, i, lr, optimizer_meta_parameter, train_loss, accu_train, test_loss, accu_test));
        end
    end
    
    %if size(samples_oos, 1) > 1
    %    [mean_test_err, test_err, pred_oos] = validate_msp_tempotron(ts, samples_oos, targets_oos, w_out, V_thresh, V_rest, tau_m, tau_s);
    %    Y_oos(k, :, 1) = targets_oos;
    %    Y_oos(k, :, 2) = pred_oos;
    %end
    
    % save model
    disp(sprintf('saving model to: %s', outputFileName));
    save(outputFileName, 'seed', 'fold', 'n_folds', 'run_folds', 'N_samples', ...
        'n_epochs', 'modelName', 'dataSetFileName', ...
        'max_class', 'min_class',...
        'losses', 'Y_train', 'Y_test', 'Y_oo', 'W', ...
        'tau_m', 'tau_s', 'V_rest', 'V_thresh', 'T', ...
        'dt', 'ts', 'lr', 'optimizer', 'optimizer_meta_parameter');

end

save(sprintf('model_cache/msp_crossval-%s-%s.mat',modelName,optimizer), ...
    'modelName', 'dataSetFileName', ...
    'max_class', 'min_class', 'optimizer', 'W_all', 'L', 'R', ...
    'Y_oos', 'n_folds', 'N_samples', 'Y', 'W');
end