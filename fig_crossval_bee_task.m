close all;
trainIdx = 1;
testIdx = 2;
eval_epoch = 8;

figure();
% plot loss TRAINING & TESTING
subplot(3,3,1);
hold on;
l1 = plot(squeeze(L(:, :,trainIdx))', 'k');
l2 = plot(squeeze(L(:, :,testIdx))', ':k');
legend([l1(1);l2(1)], {'train',' test'});
xlabel('epochs');
ylabel('error');
ylim([0 8]);
title(optimizer);

% plot RMSE & accuracy
subplot(3,3,2);
hold on;
l_rmse = plot(squeeze(R(:, :,trainIdx+2))', '-b');
plot(squeeze(R(:, :,testIdx+2))', ':b');
ylabel('RMSE');
ylim([0 8]);

yyaxis right;
hold on;
l_accu = plot(squeeze(R(:, :,trainIdx))', '-m');
plot(squeeze(R(:, :,testIdx))', ':m');
legend( [l_rmse(1);l_accu(1)] , {'RMSE','accuracy'} );
xlabel('epochs');
ylabel('accuracy [%]');
ylim([0 100]);


lst = dir(sprintf('model_cache/msp_%s-%s.*.mat', modelName, optimizer));
Y_hat = [];
Y_ = [];
[~,epoch_idx] = max(squeeze(R(:,:,2)), [], 2);
for k=1:length(lst)
   data = load(sprintf('model_cache/%s', lst(k).name), 'Y_train', 'Y_test'); 
   Y_hat = [Y_hat; squeeze(data.Y_train(epoch_idx(k), :, :))];
   Y_ = [Y_; reshape(data.Y_test(epoch_idx(k), :, :), [size(data.Y_test,2) size(data.Y_test,3)])];
end

% plot conf mat for TRAIN set
subplot(3,3, [4 7]);
hold on;
title('confusion matrix (train)');
C = confusionmat(Y_hat(:,1),Y_hat(:,2));
plotConfMat(C, unique(Y_hat));


% plot conf mat for TEST set
subplot(3,3,[5 8]);
title('confusion matrix (test)');
hold on;
%Y_ = Y(:,eval_epoch,:,:);
%Y_ = reshape(Y_, [size(Y_,1)*size(Y_,3), 2]);

C = confusionmat(Y_(:,1),Y_(:,2));
plotConfMat(C, unique(Y_));

N_iter = 1000;
unique_labels = unique(Y_(:,1));
M = zeros(length(unique_labels)-1,N_iter,2);
for k=1:size(M,1)
for i=1:N_iter
        s = randsample(unique_labels(1:k+1), 2);
        ia = randsample(find(Y_(:,1) == s(1)),1);
        ib = randsample(find(Y_(:,1) == s(2)),1);
        if Y_(ia,2) == Y_(ib,2)
            M(k,i,2) = randsample([0 1], 1);
        else
            M(k,i,2) = (s(1) > s(2) && Y_(ia,2) > Y_(ib,2)) || (s(1) < s(2) && Y_(ia,2) < Y_(ib,2));
        end
        s = randsample(unique(Y_hat(:,1)), 2);
        ia = randsample(find(Y_hat(:,1) == s(1)),1);
        ib = randsample(find(Y_hat(:,1) == s(2)),1);
        
        if Y_hat(ia,2) == Y_hat(ib,2)
            M(k,i,1) = randsample([0 1], 1);
        else
            M(k,i,1) = (s(1) > s(2) && Y_hat(ia,2) > Y_hat(ib,2)) || (s(1) < s(2) && Y_hat(ia,2) < Y_hat(ib,2));
        end
end
end
    
% experimental / pair test
subplot(3,3,[3 6 9]);
title('"greater than" task');
hold on;
%bar(mean(squeeze(M(:,:,1)), 2));
y_bar = [mean(squeeze(M(:,:,1)), 2) mean(squeeze(M(:,:,2)), 2)]' * 100;
y_bar_err = [std(squeeze(M(:,:,1)), [], 2) std(squeeze(M(:,:,2)), [], 2)]' * 100;
h = bar(y_bar, 'FaceColor', 'flat');
ngroups = size(y_bar, 1);
nbars = size(y_bar, 2);
% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, y_bar(:,i), y_bar_err(:,i), zeros(ngroups, 1), '.', 'Color', [0 0 0]);
end
for i=1:length(h)
    h(i)
end
%x_bar = categorical({'train','test'});
%h1 = bar(x_bar,mean(squeeze(M(end,:,:)),1) * 100, 'FaceColor', 'm');
%err = errorbar(x_bar, mean(squeeze(M(end,:,:)),1)*100, var(squeeze(M(end,:,:)),[],1)*100, [0 0], '.'); 
%err.Color = [0 0 0];    
%h1.FaceColor = 'flat';
%h1.CData(1,:) = [255 127 80] / 255;
%h1.CData(2,:) = [0 255 255] / 255;
ylim([0 100]);
ylabel('accuracy [%]');
set(gca, 'XTick',1:ngroups,...
    'XTickLabel', {'Train' 'Test'});
legend(num2str([zeros(length(unique_labels)-1,1)+unique_labels(1) unique_labels(2:end)], 'items %d-%d'));

set(gcf, 'Position', get(0, 'Screensize'));
print(sprintf('figures/bee_task/crossval_%s-%s.png', modelName, optimizer), '-dpng'); 

set(gcf,'PaperUnits','centimeters');
set(gcf, 'PaperSize', [50 15]);
print(sprintf('figures/bee_task/crossval_%s-%s.pdf', modelName, optimizer), '-fillpage', '-dpdf'); 