import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_visualization_q(data, target, figsize=(12, 5), bins=50):
    print('Visualization of quantitative features distribution: \n')

    for i in data.drop(target, axis=1):
        if data[i].dtype != 'O' and list(data[i].unique()) != [0, 1]:
            figure, (ax_box, ax_hist) = plt.subplots(2, 1, sharex=True,
                                                     gridspec_kw={'height_ratios': (.20, .80)}, figsize=figsize)

            sns.despine()

            sns.boxplot(data[i], ax=ax_box, orient='h').set_title(i, y=1.5, fontsize=14)
            sns.histplot(data[i], bins=bins, kde=True, ax=ax_hist)

            plt.show()
            print('Feature statistics:')
            display(data[i].describe())


def create_visualization_cat(data, target, figsize=(12, 5)):
    print('Visualization of categorical features and target prevalence:\n')

    for i in data.drop(target, axis=1):
        if data[i].dtype == 'O' or list(data[i].unique()) == [0, 1]:
            bar_data = data[i].value_counts().reset_index().sort_values(by=i)
            prev_data = data.groupby(i, as_index=False)[target].mean().sort_values(by=i)

            figure, (ax_bar, ax_prev) = plt.subplots(1, 2, figsize=figsize)

            sns.barplot(
                data=bar_data,
                x=bar_data[i], y=bar_data['count'],
                ax=ax_bar
            ).set_title(i, y=1.02, fontsize=14)

            sns.barplot(
                data=prev_data,
                x=prev_data[i], y=prev_data[target],
                ax=ax_prev
            ).set_title(f'{i}. Target prevalence', y=1.02, fontsize=14)

            plt.show()
            print('Feature statistics:')
            display(data[i].describe())


def create_visualization_target(data, target, figsize=(7, 5)):
    print('Visualization of target distribution:\n')

    plt.figure(figsize=figsize)

    sns.barplot(
        data=data[target].value_counts().reset_index(),
        x=data[target], y='count'
    ).set_title(target, fontsize=14)

    plt.show()
    print('Target statistics:')
    display(data[target].describe())


def get_corr_map(data, method='pearson', figisze=(15, 12)):
    plt.figure(figsize=figisze)

    sns.heatmap(
        round(data.corr(method=method), 2), vmax=1, vmin=-1, square=True, linewidths=3, annot=True, cmap='coolwarm'
    )

    plt.show()


def create_metrics(model, features, target, only_result):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
        roc_auc_score, precision_recall_curve, roc_curve

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    results = pd.DataFrame(
        {
            'accuracy': [accuracy_score(target, predictions)],
            'precision': [precision_score(target, predictions)],
            'recall': [recall_score(target, predictions)],
            'f1': [f1_score(target, predictions)],
            'auc': [roc_auc_score(target, predictions)],
        }
    ).T.reset_index().rename(columns={'index': 'metrics', 0: 'score'})

    if only_result:
        return results

    display(results)

    conf_matrix = pd.DataFrame(confusion_matrix(target, predictions))
    conf_matrix_norm = pd.DataFrame(confusion_matrix(target, predictions, normalize='true') * 100)

    display(conf_matrix)

    figure, (ax_roc, ax_f1, ax_matrix) = plt.subplots(1, 3, figsize=(21, 5))

    fpr, tpr, thresholds = roc_curve(target, probabilities)
    ax_roc.plot(fpr, tpr, lw=2, label='ROC curve')
    ax_roc.plot([0, 1], [0, 1])
    ax_roc.set_xlim([-0.05, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC curve')

    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    ax_f1.step(recall, precision, where='post')
    ax_f1.set_xlim([-0.05, 1.0])
    ax_f1.set_ylim([0.0, 1.05])
    ax_f1.set_xlabel('Recall')
    ax_f1.set_ylabel('Precision')
    ax_f1.set_title('Precision-Recall')

    sns.heatmap(
        conf_matrix_norm,
        linewidths=3,
        linecolor='white',
        annot=True,
        cmap='Blues',
        ax=ax_matrix
    )
    ax_matrix.set_title('Confusion Matrix')

    plt.show()


def create_cv(model, features, target, cat_features, folds, random_state):
    from catboost import Pool, cv

    params = {
        'random_state': random_state,
        'use_best_model': model.get_params()['use_best_model'],
        'auto_class_weights': model.get_params()['auto_class_weights'],
        'early_stopping_rounds': 150,
        'loss_function': 'Logloss',
        'custom_loss': 'AUC',
        'iterations': model.get_params()['iterations'],
        'depth': model.get_params()['depth'],
        'learning_rate': model.get_params()['learning_rate'],
        'verbose': False
    }

    cv_ds = Pool(
        data=features,
        label=target,
        cat_features=cat_features
    )
    scores = cv(
        cv_ds,
        params,
        fold_count=folds,
        plot=True
    )
    return scores


def get_features_importances(model, features_train):
    if model.feature_importances_ != ():
        feature_importances = pd.Series(
            model.feature_importances_,
            index=features_train.columns
        ).reset_index().rename(
            columns={'index': 'feature', 0: 'importance'}
        ).sort_values(
            by='importance',
            ascending=False
        ).reset_index(drop=True)

        feature_importances['feature'] = feature_importances['feature'].apply(str)

        plt.figure(figsize=(10, 10))
        sns.barplot(y=feature_importances['feature'], x=feature_importances['importance'])
        plt.show()

    else:
        print('Can not calculate feature importances for loaded model')


def get_shap(model, features, target, plot_size=(12, 10)):
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features, target)

    shap.summary_plot(shap_values, features, plot_size=plot_size)