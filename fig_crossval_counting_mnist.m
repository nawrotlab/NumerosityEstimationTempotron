close all;
trainIdx = 1;
testIdx = 2;
eval_epoch = 6;

figure();
% plot train/test convergence
subplot(3,3,1);
hold on;
convNet = load("model_cache/CNN_mnist_grid_ones_0-5");
tmp = mean(convNet.data.R(:, 1:end-6, 3), 1) * 100;
x = [0 convNet.data.epoch_list(1:end-6)];
l1 = plot(x, [(1/7*100) tmp], '-o', 'Color', [.5 .5 .5], ...
    'MarkerSize', 2, 'MarkerFaceColor', [.5 .5 .5]);
plot([0], [1/7] * 100, 's', 'Color', [.5 .5 .5], 'MarkerFaceColor', [.5 .5 .5]);

disp(sprintf('epoch: %d | %.4f', convNet.data.epoch_list(end-3), mean(convNet.data.R(:, end-3, 3), 1) * 100));
plot([35], mean(convNet.data.R(:, end-3, 3), 1) * 100, 'o', ...
    'Color', [.5 .5 .5], 'MarkerFaceColor', [.5 .5 .5]);

legend([], {'validation'});
xlabel('epochs');
ylabel('accuracy [%]');
ylim([0 100]);
xlim([-1 40]);
xticks([1 5 10 15 20 25 30 35]);
xticklabels(sprintfc('%d', [1 5 10 15 20 25 30 200]));
title('ConvNet');
%l1 = plot(squeeze(L(:, :,trainIdx))', 'k');
%l2 = plot(squeeze(L(:, :,testIdx))', ':k');
%legend([l1(1);l2(1)], {'train',' test'});
%xlabel('epochs');
%ylabel('error');
%ylim([0 8]);
%title(optimizer);

% plot RMSE & accuracy
subplot(3,3,2);
hold on;
tmp = mean(squeeze(R(:, :,testIdx))',2);
l1 = plot(0:size(tmp), [(1/7*100); tmp], '-o', 'Color', [.5 .5 .5], ...
    'MarkerSize', 2, 'MarkerFaceColor', [.5 .5 .5]); %test accuracy
tmp = mean(squeeze(R(:, :,trainIdx))',2);
l2 = plot(0:size(tmp), [(1/7*100); tmp], '-o', 'Color', [0 0 0], ...
    'MarkerSize', 2, 'MarkerFaceColor', [0 0 0]); % train accuracy
plot([0], [1/7] * 100, 's', 'Color', [0 0 0], 'MarkerFaceColor', [0 0 0]);
legend([l1, l2], {'validation', 'training'});
xlabel('epochs');
ylabel('accuracy [%]');
ylim([0 100]);
xlim([-1 size(R,2)]);
xticks([1 5 10 15 20 25 30]);
%l_rmse = plot(squeeze(R(:, :,trainIdx+2))', '-b');
%plot(squeeze(R(:, :,testIdx+2))', ':b');
%ylabel('RMSE');
%ylim([0 8]);
%yyaxis right;
%hold on;
%l_accu = plot(squeeze(R(:, :,trainIdx))', '-m');
%plot(squeeze(R(:, :,testIdx))', ':m');
%legend( [l_rmse(1);l_accu(1)] , {'RMSE','accuracy'} );
%xlabel('epochs');
%ylabel('accuracy [%]');
%ylim([0 100]);


lst = dir(sprintf('model_cache/msp_%s-%s.*.mat', modelName, optimizer));
Y_hat = [];
Y_ = [];
Y_outofsamples = [];
Y_size = zeros(length(lst),3);
[~,epoch_idx] = max(squeeze(R(:,:,2)), [], 2);

rmse_mu = mean(diag(R(:,epoch_idx,testIdx+2)), 1);
rmse_var = std(diag(R(:,epoch_idx,testIdx+2)), [], 1);
accu_mu = mean(diag(R(:,epoch_idx,testIdx)), 1);
accu_var = std(diag(R(:,epoch_idx,testIdx)), [], 1);
disp(sprintf('Avg. RMSE: %.4f (%.4f) | Avg. Accuracy: %.4f (%.4f)', rmse_mu, rmse_var, accu_mu, accu_var));
for k=1:length(lst)
   data = load(sprintf('model_cache/%s', lst(k).name), 'Y_train', 'Y_test', 'Y_oo'); 
   Y_size(k,1) = size(data.Y_train,2);
   Y_size(k,2) = size(data.Y_test,2);
   Y_size(k,3) = size(data.Y_oo,2);
   Y_hat = [Y_hat; squeeze(data.Y_train(epoch_idx(k), :, :))];
   Y_ = [Y_; reshape(data.Y_test(epoch_idx(k), :, :), [size(data.Y_test,2) size(data.Y_test,3)])];
   Y_outofsamples = [Y_outofsamples; reshape(data.Y_oo(epoch_idx(k), :, :), [size(data.Y_oo,2) size(data.Y_oo,3)])];
end


% plot accuracy per category
subplot(3,3,3);
hold on;

% plot results for targets > n_classes
accu_means = [];
accu_std = [];
oos_classes = unique(squeeze(Y_outofsamples(:,1)));
oos_n_classes = length(oos_classes);
oos_max_class = max(oos_classes);
oos_min_class = min(oos_classes);
accu_per_class = zeros(oos_n_classes, size(Y_oos,1));

for i=1:length(oos_classes)
    start = 1;
    for m=1:size(Y_size,1)
        stop = start + (Y_size(m,3)-1); %min(, Y_size(m,1));
        idx = find(Y_outofsamples(start:stop,1)==oos_classes(i));
        err = squeeze(abs(Y_outofsamples(idx,1)-Y_outofsamples(idx,2)));
        accu_per_class(i,m) = (length(find(err == 0)) * 100)/length(idx);
        start = stop + 1;
    end
end


accu_per_class_oos = accu_per_class;

n_classes = length(unique(Y_hat(:,1)));
accu_per_class = zeros(max_class+1, size(Y_size,1));
chance_level = 1/n_classes;

for i=1:max_class+1
    start = 1;
    for m=1:size(Y_size,1)
        stop = start + (Y_size(m,2)-1);
        idx = find(Y_hat(start:stop,1)==(i-1));
        err = squeeze(abs(Y_hat(idx,1)-Y_hat(idx,2)));
        accu_per_class(i,m) = (length(find(err == 0)) * 100)/length(idx);
        start = stop + 1;
    end
end

accu_per_class = [accu_per_class; accu_per_class_oos];

h1 = bar(0:size(accu_per_class,1)-1,mean(accu_per_class,2), 'FaceColor', 'm');
err = errorbar(0:size(accu_per_class,1)-1,mean(accu_per_class,2), zeros(size(accu_per_class,1),1), std(accu_per_class,[],2), '.'); 
err.Color = [0 0 0];                            
l = plot([-0.5, size(accu_per_class,1)-0.5], [1/size(accu_per_class,1), 1/size(accu_per_class,1)]*100, '-.k'); 
%xticks(min(Y_hat(:,2)):max([max_class max(Y_hat(:,2))]));
xlabel('category / #spikes');
ylabel('accuracy [%]');
ylim([0 100]);
xlim([-0.5, size(accu_per_class,1)-0.5]);
title('trained distribution');
ax1 = gca;
legend([l, h1], {'chance level', 'accuracy'}, 'Location', 'northeast');


subplot(3,3,[4 5 7 8]);
title('confusion matrix');
hold on;
%C = confusionmat(Y_(idx,1),Y_(idx,2), 'Order', unique(Y_(idx,2)));
C = confusionmat(Y_(:,1),Y_(:,2));
%C = C(~all(C==0,2), ~all(C==0,2)); % drop all null rows/cols
%if size(C,1) ~= max(Y_(idx,1))
%   C(2:end - (max(Y_(idx,2)) - max(Y_(idx,1))), :) 
%end
plotConfMat(C, unique(Y_));
%if size(C,1) ~= max(Y_(:,1))
%    ylim([0 length(unique(Y_(:,1)))+1.5]);
%    xlim([0 length(unique(Y_(:,1)))+1.5]);
%end
%modelName = 'counting-mnist-grid-focal-evenodd';

set(gcf, 'Position', get(0, 'Screensize'));
print(sprintf('figures/counting-mnist-grid/crossval_%s-%s.png', modelName, optimizer), '-dpng'); 