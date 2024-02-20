def get_trained_model(features_train, target_train, features_test, target_test, cat_features, random_state=68,
                      task_type='CPU', grid_search=False, save=False):
    from catboost import CatBoostClassifier

    if grid_search:
        cat_model = CatBoostClassifier(
            random_state=random_state,
            auto_class_weights='Balanced',
            early_stopping_rounds=100,
            verbose=False,
            boosting_type='Ordered',
            cat_features=cat_features,
            task_type=task_type
        )

        grid = {
            'depth': list(range(4, 8)),
            'l2_leaf_reg': list(range(2, 6)),
            'learning_rate': [0.001, 0.003, 0.005],
            'iterations': [3000, 5000, 10000]
        }

        search_res = cat_model.grid_search(
            grid,
            features_train,
            target_train,
            calc_cv_statistics=True,
            verbose=False,
            plot=True,
        )

        cat_model = CatBoostClassifier(
            random_state=random_state,
            auto_class_weights='Balanced',
            early_stopping_rounds=100,
            verbose=False,
            boosting_type='Ordered',
            cat_features=cat_features,

            task_type=task_type,

            use_best_model=True,
            depth=search_res['params']['depth'],
            l2_leaf_reg=search_res['params']['l2_leaf_reg'],
            learning_rate=search_res['params']['learning_rate'],
            iterations=search_res['params']['iterations']
        )

    else:
        cat_model = CatBoostClassifier(
            random_state=random_state,
            auto_class_weights='Balanced',
            early_stopping_rounds=100,
            verbose=False,
            boosting_type='Ordered',
            cat_features=cat_features,
            task_type=task_type,
            use_best_model=True,

            depth=5,
            l2_leaf_reg=4,
            learning_rate=0.003,
            iterations=3000
        )

    cat_model.fit(features_train, target_train, eval_set=(features_test, target_test), verbose=False, plot=True)

    if save:
        import datetime

        now = datetime.datetime.now()
        score_model = round(cat_model.score(features_test, target_test), 4) * 1000
        path = f'.\models\model-{score_model} - {now.strftime("%Y-%m-%d-%H-%M")}.json'

        cat_model.save_model(path, format='json')

    return cat_model

