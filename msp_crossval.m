function [results] = msp_crossval(modelName, dataSetFileName, varargin)

args = inputParser;
defaultOptimizer='rmsprop';
validOptimizers = {'rmsprop', 'momentum'};
checkOptimizer = @(x) any(validatestring(x,validOptimizers));

addRequired(args,'modelName',@ischar);
addRequired(args,'dataSetFileName',@ischar);
%addParameter(args, 'n_samples', -1, @isnumeric);
addParameter(args, 'n_epochs', 10, @isnumeric);
addParameter(args, 'n_models', 1, @isnumeric);
addParameter(args, 'dt', 1/1000, @isnumeric);
addParameter(args, 'optimizer', defaultOptimizer, checkOptimizer);
addParameter(args, 'rng_seed', 42, @isnumeric);
addParameter(args, 'n_folds', 10, @isnumeric);
addParameter(args, 'lr', 0.001, @isnumeric);
addParameter(args, 'rmsprop_gamma', 0.9, @isnumeric);

args.KeepUnmatched = true;
parse(args,modelName, dataSetFileName, varargin{:});

%n_samples = args.Results.n_samples;
n_models = args.Results.n_models;
n_epochs = args.Results.n_epochs;
optimizer = args.Results.optimizer;
seed = args.Results.rng_seed;
n_folds = args.Results.n_folds;
lr = args.Results.lr;
rmsprop_gamma = args.Results.rmsprop_gamma;
dt = args.Results.dt;

rng(seed);
seeds = randi(98756, 1, n_models);

load(dataSetFileName);

N_trials = size(data.samples, 1);
N_syn = size(data.samples, 2);

try
   T = double(data.T_trial);     % trial duration in sec
catch
   T = double(ceil((data.imageWidth * data.T_pixel) + 0.05));     % trial duration in sec
end

ts = 0:dt:T;

% neuron model
tau_m = 0.015;
tau_s = 0.005;
V_thresh = 1;
V_rest = 0;

X = data.samples;
y = double(data.rewards);
results = [];

for m=1:n_models
    rng(seeds(m));
    c = cvpartition(uint16(y),'kfold', n_folds);
    N_sim = c.NumTestSets;

    mean_losses = zeros(2, c.NumTestSets);
    
for k=1:c.NumTestSets
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % run learning vor several epochs
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        train_losses = [];
        validation_losses= [];
        weights_per_epoch = [];
        w_init = normrnd(0, 1 / N_syn, 1, N_syn);
        
        n_iter = 0;
        rng(seed);
        train_idx = c.training(k);
        test_idx = c.test(k);
        
        X_train = X(train_idx, :);
        y_train = y(train_idx);
        X_test = X(test_idx, :);
        y_test = y(test_idx);
        
        w_out = w_init;
        predictions = {};
        train_losses = zeros(1, n_epochs);
        validation_losses = zeros(1, n_epochs);
        weights_per_epoch = zeros(N_syn, n_epochs);

        for i=1:n_epochs
            [w_out, ~, ~, errs, y_pred, ~, ~, n_iter] = fit_msp_tempotron(ts, X_train, y_train, w_out, V_thresh, V_rest, tau_m, tau_s, lr, n_iter, optimizer);
            loss = mean(abs(y_pred-y_train));
            train_losses(1,i) = loss;
            weights_per_epoch(:, i) = w_out;
            
            [mean_val_loss, ~, y_pred_test] = validate_msp_tempotron(ts, X_test, y_test, w_out, V_thresh, V_rest, tau_m, tau_s);
            validation_losses(1,i) = mean_val_loss;
            n_correct = length(find(y_pred_test==y_test));
            n_total = length(y_test);
            
            predictions{end+1} = {y_train, y_pred, y_test, y_pred_test};
            
            if (isempty(errs)) % all zeros => no errors, converged
                disp(sprintf('%s learning converged after %d epochs', optimizer, i));
                break;
            end

            if (mod(i, 1) == 0)
                disp(sprintf('%d [%d/%d] %s | epoch=%d | lr=%.4f | train_loss: %.3f | val_loss: %.3f (%d/%d)', m, k, c.NumTestSets, optimizer, i, lr, loss, mean_val_loss, n_correct, n_total));
            end
        end


    outputFileName = sprintf('model_cache/msp_%s.%s.%d.fold.%d.mat', modelName, optimizer, m, k);
    disp(sprintf('saving model to: %s', outputFileName));
    save(outputFileName, 'k', 'seed', 'dataSetFileName', 'train_idx', 'test_idx', 'predictions', 'train_losses', 'validation_losses', 'weights_per_epoch', 'w_out', 'w_init', 'tau_m', 'tau_s', 'V_rest', 'V_thresh', 'T', 'dt', 'ts', 'lr', 'n_epochs');

    mean_losses(1, k) = mean(train_losses);
    mean_losses(2, k) = mean(validation_losses);
end

results = [results; mean_losses];
end
end