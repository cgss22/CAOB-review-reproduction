# -*- coding: utf-8 -*-
"""Merged original project source. Algorithms are retained, including legacy defects.
No old Python scripts are executed or read at runtime. Original datasets/models are required.
Outputs mirror original relative paths under Revised JGE/fig/_original_reproduction.
Default frozen workflow uses existing inputs; regeneration is a separate workflow.
"""
from pathlib import Path
import os, json, traceback, gc
PROJECT_ROOT = Path(os.environ.get('CAOB_PROJECT', Path(__file__).resolve().parent.parent))
ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT / 'data'
OUT = Path(os.environ.get('CAOB_OUTPUT', ROOT / 'Revised JGE' / 'result' / 'CAOB_Review' / 'Original_Analysis'))
ACTIVE_STAGE = ''

def _output(value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return value
    p = Path(os.fsdecode(value))
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if p.parent == OUT.resolve():
        return str(p)
    import hashlib
    suffix = hashlib.sha256(str(p.with_suffix('')).encode('utf8')).hexdigest()[:8]
    return str(OUT / (ACTIVE_STAGE + '__' + p.stem + '__' + suffix + p.suffix))

def _star_import(module_name):
    import importlib
    module = importlib.import_module(module_name)
    names = getattr(module, '__all__', [n for n in vars(module) if not n.startswith('_')])
    globals().update({name: getattr(module, name) for name in names})

def original_000():
    """
Created on Fri Oct 11 16:11:59 2024

@author: ASUS
"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Flatten, Dense
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from sklearn.model_selection import train_test_split
    import os
    train_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/PCA_with_labels_Train.csv'
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/PCA_with_labels_Test.csv'
    pca_data_train = pd.read_csv(train_file_path)
    pca_data_test = pd.read_csv(test_file_path)
    pca_columns = [f'PCA{i}' for i in range(1, 24)]
    train_pca_values = pca_data_train[pca_columns].values
    train_labels = pca_data_train['Label'].values
    train_image = train_pca_values.reshape(-1, 23, 1)
    train_label_image = train_labels
    input_shape = (23, 1)
    input_combined = Input(shape=input_shape)
    x = Conv1D(64, 3, padding='same', activation='relu')(input_combined)
    x = BatchNormalization()(x)
    x = Conv1D(32, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    final_output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_combined, outputs=final_output)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=1e-06)
    checkpoint_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/best_model_full_image_1DCNN_test.keras'
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='accuracy', save_best_only=True)
    history = model.fit(train_image, train_label_image, batch_size=32, epochs=11, verbose=2)
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test'
    if not os.path.exists(output_dir):
        os.makedirs(str(OUT), exist_ok=True)
    model_save_path = os.path.join(output_dir, 'trained_model_1DCNN_test.h5')
    model.save(_output(model_save_path))
    print(f'训练好的模型已保存至: {model_save_path}')
    test_pca_values = pca_data_test[pca_columns].values
    test_image = test_pca_values.reshape(-1, 23, 1)
    predictions = model.predict(test_image)
    pca_data_test['Predicted_Probability'] = predictions.flatten()
    output_file = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/PCA_with_full_predictions_Test_1DCNN_test.csv'
    pca_data_test.to_csv(_output(output_file), index=False)
    print(f'完整预测结果已保存至 {output_file}')
    x_coords = pca_data_test['x'].values
    y_coords = pca_data_test['y'].values
    predicted_probs = pca_data_test['Predicted_Probability'].values
    plt.figure(figsize=(18, 5.64), dpi=100)
    plt.tricontourf(x_coords, y_coords, predicted_probs, levels=14, cmap='coolwarm')
    plt.colorbar(label='Predicted Probability')
    plt.axis('off')
    plt.title('Predicted Probability Contour Map 1DCNN')
    plt.savefig(_output('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/contour_map_1DCNN_test.png'), dpi=100, bbox_inches='tight')
    plt.savefig(_output('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/contour_map_1DCNN_test.svg'), dpi=100, bbox_inches='tight')
    plt.show()
    plt.style.use('default')
    plt.plot(history.history['accuracy'], label='Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.savefig(_output('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/accuracy_curve_1DCNN_test.png'), dpi=100, bbox_inches='tight')
    plt.savefig(_output('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/accuracy_curve_1DCNN_test.svg'), dpi=100, bbox_inches='tight')
    plt.show()
    plt.plot(history.history['loss'], label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.savefig(_output('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/loss_curve_1DCNN_test.png'), dpi=100, bbox_inches='tight')
    plt.savefig(_output('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/loss_curve_1DCNN_test.svg'), dpi=100, bbox_inches='tight')
    plt.show()
    plt.scatter(range(len(history.history['accuracy'])), history.history['accuracy'], label='Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([min(history.history['accuracy']) - 0.05, max(history.history['accuracy']) + 0.05])
    plt.yticks([round(i * 0.1, 2) for i in range(7, 11)])
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.title('Training Accuracy Curve')
    plt.savefig(_output('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/accuracy_curve_1DCNN_scatter_1DCNN_test.png'))
    plt.savefig(_output('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/accuracy_curve_1DCNN_scatter_1DCNN_test.svg'), format='svg')
    plt.show()
    plt.scatter(range(len(history.history['loss'])), history.history['loss'], label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([min(history.history['loss']) - 0.05, max(history.history['loss']) + 0.05])
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.title('Training Loss Curve')
    plt.savefig(_output('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/loss_curve_1DCNN_scatter_1DCNN_test.png'))
    plt.savefig(_output('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/loss_curve_1DCNN_scatter_1DCNN_test.svg'), format='svg')
    plt.show()
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test'
    if not os.path.exists(output_dir):
        os.makedirs(str(OUT), exist_ok=True)

    def pca_feature_sensitivity_analysis(model, test_patches, feature_index):
        """
    对特定PCA特征进行敏感性分析，通过屏蔽单个特征并观察模型输出的变化。
    
    Args:
    - model: 训练好的Keras模型
    - test_patches: 测试集中的数据块
    - feature_index: 要屏蔽的特征索引（0-22之间）
    
    Returns:
    - predictions_diff: 屏蔽特征前后的预测值差异
    """
        modified_patches = np.copy(test_patches)
        modified_patches[:, feature_index, :] = 0
        original_predictions = model.predict(test_patches)
        modified_predictions = model.predict(modified_patches)
        predictions_diff = np.abs(original_predictions - modified_predictions)
        return predictions_diff
    feature_contributions = []
    for i in range(23):
        contribution = pca_feature_sensitivity_analysis(model, test_image, i)
        feature_contributions.append(np.mean(contribution))
    plt.figure(figsize=(10, 6))
    plt.bar(range(23), feature_contributions, color='skyblue')
    plt.xlabel('PCA Feature Index')
    plt.ylabel('Average Contribution to Prediction')
    plt.title('PCA Feature Sensitivity Analysis')
    output_png_path = os.path.join(output_dir, 'pca_feature_contributions_1DCNN_test.png')
    output_svg_path = os.path.join(output_dir, 'pca_feature_contributions_1DCNN_test.svg')
    plt.savefig(_output(output_png_path), format='png', dpi=100, bbox_inches='tight')
    plt.savefig(_output(output_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    print(f'特征贡献度图已保存为: {output_png_path} 和 {output_svg_path}')

def original_001():
    """
Created on Fri Oct 11 20:41:07 2024

@author: ASUS
"""
    '\nCreated on Fri Oct 11 16:11:59 2024\n\n@description: 加载训练好的1D CNN模型，并对各个PCA特征进行敏感性分析\n'
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    import os
    model_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/ensure/trained_model_1DCNN_test.h5'
    model = load_model(model_path)
    print(f'模型已成功加载: {model_path}')
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/PCA_with_labels_Test.csv'
    pca_data_test = pd.read_csv(test_file_path)
    pca_columns = [f'PCA{i}' for i in range(1, 24)]
    test_pca_values = pca_data_test[pca_columns].values
    test_image = test_pca_values.reshape(-1, 23, 1)

    def pca_feature_sensitivity_analysis(model, test_patches, feature_index):
        """
    对特定PCA特征进行敏感性分析，通过屏蔽单个特征并观察模型输出的变化。
    
    Args:
    - model: 训练好的Keras模型
    - test_patches: 测试集中的数据块
    - feature_index: 要屏蔽的特征索引（0-22之间）
    
    Returns:
    - predictions_diff: 屏蔽特征前后的预测值差异
    """
        modified_patches = np.copy(test_patches)
        modified_patches[:, feature_index, :] = 0
        original_predictions = model.predict(test_patches)
        modified_predictions = model.predict(modified_patches)
        predictions_diff = np.abs(original_predictions - modified_predictions)
        return predictions_diff
    feature_contributions = []
    for i in range(23):
        contribution = pca_feature_sensitivity_analysis(model, test_image, i)
        feature_contributions.append(np.mean(contribution))
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test'
    if not os.path.exists(output_dir):
        os.makedirs(str(OUT), exist_ok=True)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(23), feature_contributions, color='skyblue', edgecolor='black')
    plt.xlabel('PCA Feature Index', fontsize=12)
    plt.ylabel('Average Contribution to Prediction', fontsize=12)
    plt.title('PCA Feature Sensitivity Analysis', fontsize=14)
    output_png_path = os.path.join(output_dir, 'pca_feature_contributions_1DCNN_test_fraze.png')
    output_svg_path = os.path.join(output_dir, 'pca_feature_contributions_1DCNN_test_fraze.svg')
    plt.savefig(_output(output_png_path), format='png', dpi=100, bbox_inches='tight')
    plt.savefig(_output(output_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    print(f'Feature contribution plots saved as: {output_png_path} and {output_svg_path}')
    probabilities = model.predict(test_image, verbose=0).ravel()
    export_prediction_map(pca_data_test, probabilities, 'PCA_1DCNN_FROZEN')
    export_pca_sensitivity(feature_contributions, 'PCA_1DCNN_FROZEN')

def original_003():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/PCA_with_labels_Test.csv'
    pca_data_test = pd.read_csv(test_file_path)
    x_coords = pca_data_test['x'].values
    y_coords = pca_data_test['y'].values
    pca_values = pca_data_test.iloc[:, -23:].values

    def prepare_test_data_by_coordinates(x_coords, y_coords, pca_values, patch_size=9, min_points=50):
        """
    按步长划分测试集为 patch_size*patch_size 的块，X 和 Y 坐标按 9 递增。
    Args:
    - x_coords: X 坐标数组
    - y_coords: Y 坐标数组
    - pca_values: PCA 特征数组
    - patch_size: 块的大小，默认为 9
    - min_points: 每个块的最小点数，默认为 50
    """
        reshaped_patches = []
        indices = []
        min_x, max_x = (min(x_coords), max(x_coords))
        min_y, max_y = (min(y_coords), max(y_coords))
        x_steps = np.arange(min_x, max_x + 1, patch_size)
        y_steps = np.arange(min_y, max_y + 1, patch_size)
        for y_min in y_steps:
            y_max = y_min + patch_size - 1
            if y_max > max_y:
                continue
            for x_min in x_steps:
                x_max = x_min + patch_size - 1
                if x_max > max_x:
                    continue
                mask = (x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)
                patch_data = pca_values[mask]
                num_points = len(patch_data)
                if num_points >= min_points:
                    if num_points < patch_size * patch_size:
                        patch_data = np.pad(patch_data, ((0, patch_size * patch_size - num_points), (0, 0)), mode='constant')
                    try:
                        reshaped_patch = patch_data.reshape(patch_size, patch_size, -1)
                        reshaped_patches.append(reshaped_patch)
                        indices.append((x_min, x_max, y_min, y_max))
                    except ValueError as e:
                        print(f'Error reshaping patch at X: {x_min}-{x_max}, Y: {y_min}-{y_max} -> {e}')
                else:
                    print(f'Skipping patch at X: {x_min}-{x_max}, Y: {y_min}-{y_max}, insufficient points.')
        print(f'Total patches generated: {len(reshaped_patches)}')
        return (np.array(reshaped_patches), indices)
    test_patches, indices = prepare_test_data_by_coordinates(x_coords, y_coords, pca_values, patch_size=9, min_points=50)
    print(f'Test patches shape: {test_patches.shape}')
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/2DCNN_test/ensure'
    if not os.path.exists(output_dir):
        os.makedirs(str(OUT), exist_ok=True)
    model_path = os.path.join(output_dir, '2DCNN_model.h5')
    model = load_model(model_path)
    print(f'模型已加载：{model_path}')

    def pca_feature_sensitivity_analysis(model, test_patches, feature_index):
        """
    对特定PCA特征进行敏感性分析，通过屏蔽单个特征并观察模型输出的变化。
    
    Args:
    - model: 训练好的Keras模型
    - test_patches: 测试集中的数据块
    - feature_index: 要屏蔽的特征索引（0-22之间）

    Returns:
    - predictions_diff: 屏蔽特征前后的预测值差异
    """
        modified_patches = np.copy(test_patches)
        modified_patches[:, :, :, feature_index] = 0
        original_predictions = model.predict(test_patches)
        modified_predictions = model.predict(modified_patches)
        predictions_diff = np.abs(original_predictions - modified_predictions)
        return predictions_diff
    feature_contributions = []
    for i in range(23):
        contribution = pca_feature_sensitivity_analysis(model, test_patches, i)
        feature_contributions.append(np.mean(contribution))
    plt.figure(figsize=(10, 6))
    plt.bar(range(23), feature_contributions, color='skyblue', edgecolor='black')
    plt.xlabel('PCA Feature Index')
    plt.ylabel('Average Contribution to Prediction')
    plt.title('PCA Feature Sensitivity Analysis')
    output_png_path = os.path.join(output_dir, 'pca_feature_contributions_new_2DCNN_test.png')
    output_svg_path = os.path.join(output_dir, 'pca_feature_contributions_new_2DCNN_test.svg')
    plt.savefig(_output(output_png_path), format='png', dpi=100, bbox_inches='tight')
    plt.savefig(_output(output_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    print(f'特征贡献度图已保存为: {output_png_path} 和 {output_svg_path}')
    probabilities = model.predict(test_patches, verbose=0).ravel()
    full = np.full(len(pca_data_test), np.nan, dtype=float)
    for k, (x_min, x_max, y_min, y_max) in enumerate(indices):
        mask = (x_coords >= x_min) & (x_coords < x_max) & (y_coords >= y_min) & (y_coords < y_max)
        full[mask] = probabilities[k]
    export_prediction_map(pca_data_test, full, 'PCA_2DCNN_FROZEN')
    pd.DataFrame(indices, columns=['x_min', 'x_max', 'y_min', 'y_max']).assign(Predicted_Probability=probabilities).to_csv(OUT / 'PCA_2DCNN_FROZEN__patch_predictions.csv', index=False)
    export_pca_sensitivity(feature_contributions, 'PCA_2DCNN_FROZEN')

def original_037():
    import numpy as np
    import pandas as pd
    import random
    import matplotlib.pyplot as plt
    pca_data = pd.read_csv('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA/PCA.csv')
    label_data_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/Sample Location_delet_orogenic_gold.xlsx'
    df = pd.read_excel(label_data_path)
    labels = np.full((564, 1800), np.nan)
    longitude_min, longitude_max = (55, 135)
    latitude_min, latitude_max = (35, 60)
    x_pixels = (df['Longitude'] - longitude_min) / (longitude_max - longitude_min) * 1799
    y_pixels = (df['Latitude'] - latitude_min) / (latitude_max - latitude_min) * 563

    def expand_to_9x9(x, y, label_layer):
        """扩展每个矿点为 9x9 的像素块，并确保不重复标记"""
        h, w = label_layer.shape
        if x - 4 < 0 or x + 4 >= w or y - 4 < 0 or (y + 4 >= h):
            return
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                new_x = int(x + dx)
                new_y = int(y + dy)
                if 0 <= new_x < w and 0 <= new_y < h:
                    if np.isnan(label_layer[new_y, new_x]):
                        label_layer[new_y, new_x] = 1
    for x, y in zip(x_pixels, y_pixels):
        expand_to_9x9(x, y, labels)

    def generate_negative_samples(label_layer, num_positive_samples, patch_size=9, margin=12):
        """生成与正样本数量相同的负样本"""
        negative_samples = []
        h, w = label_layer.shape

        def is_valid_negative(x, y, label_layer, negative_samples, margin):
            """检查当前随机生成的负样本是否有效"""
            if x - patch_size // 2 - margin < 0 or x + patch_size // 2 + margin >= w:
                return False
            if y - patch_size // 2 - margin < 0 or y + patch_size // 2 + margin >= h:
                return False
            for dx in range(-patch_size // 2, patch_size // 2 + 1):
                for dy in range(-patch_size // 2, patch_size // 2 + 1):
                    if label_layer[y + dy, x + dx] == 1:
                        return False
            for neg_x, neg_y in negative_samples:
                if abs(neg_x - x) < patch_size and abs(neg_y - y) < patch_size:
                    return False
            return True
        while len(negative_samples) < num_positive_samples:
            x = random.randint(patch_size // 2 + margin, w - patch_size // 2 - margin - 1)
            y = random.randint(patch_size // 2 + margin, h - patch_size // 2 - margin - 1)
            if is_valid_negative(x, y, label_layer, negative_samples, margin):
                negative_samples.append((x, y))
        return negative_samples
    num_positive_samples = len(x_pixels)
    negative_samples = generate_negative_samples(labels, num_positive_samples)
    for neg_x, neg_y in negative_samples:
        expand_to_9x9(neg_x, neg_y, labels)
    pca_data['Label'] = np.nan
    for x, y in zip(x_pixels, y_pixels):
        x_pixel = int(x)
        y_pixel = int(y)
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                new_x = x_pixel + dx
                new_y = y_pixel + dy
                if 0 <= new_x < 1800 and 0 <= new_y < 564:
                    pca_data.loc[(pca_data['X'] == new_x) & (pca_data['Y'] == new_y), 'Label'] = 1
    for neg_x, neg_y in negative_samples:
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                new_x = int(neg_x + dx)
                new_y = int(neg_y + dy)
                if 0 <= new_x < 1800 and 0 <= new_y < 564:
                    pca_data.loc[(pca_data['X'] == new_x) & (pca_data['Y'] == new_y), 'Label'] = 0
    output_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA_with_labels.csv'
    pca_data.to_csv(_output(output_file_path), index=False)
    print(f'新标签数据已保存到 {output_file_path}')
    pca_data = pd.read_csv('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA_with_labels.csv')
    positive_samples = pca_data[pca_data['Label'] == 1]
    negative_samples = pca_data[pca_data['Label'] == 0]
    nan_samples = pca_data[pca_data['Label'].isna()]
    fig, ax = plt.subplots(figsize=(18, 5.64), dpi=100)
    background = np.ones((564, 1800))
    ax.imshow(background, cmap='binary', interpolation='none', origin='upper')
    ax.scatter(positive_samples['X'], positive_samples['Y'], color='red', s=30, label='矿点 (正样本)', marker='o')
    ax.scatter(negative_samples['X'], negative_samples['Y'], color='blue', s=30, label='非矿点 (负样本)', marker='x')
    ax.set_xlim([0, 1799])
    ax.set_ylim([0, 563])
    ax.legend(loc='upper right')
    ax.set_title('矿点、非矿点与未定义区域分布图', fontsize=16)
    ax.axis('off')
    output_image_path_png = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/complete_mining_and_non_mining_samples.png'
    output_image_path_svg = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/complete_mining_and_non_mining_samples.svg'
    plt.savefig(_output(output_image_path_png), dpi=100, bbox_inches='tight')
    plt.savefig(_output(output_image_path_svg), format='svg', bbox_inches='tight')
    plt.show()
    print(f'矿点、非矿点与未定义区域的分布图已保存到 {output_image_path_png} 和 {output_image_path_svg}')

def original_042():
    """
Created on Wed Nov 13 11:02:31 2024

@author: ASUS
"""
    import pandas as pd
    import os
    file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Data_analyses/Date_Th_U/output_with_best_age_and_era_labels.xlsx'
    df = pd.read_excel(file_path)
    df = df[['Th/U', 'Best age']]
    df['Age Group'] = df['Best age'] // 10 * 10
    age_groups = df.groupby('Age Group')['Th/U'].mean().reset_index()
    output_file_path = os.path.splitext(file_path)[0] + '_age_group_average.xlsx'
    age_groups.to_excel(_output(output_file_path), index=False)
    print(f'结果已保存到 {output_file_path}')

def original_043():
    """
Created on Thu Nov 14 10:47:15 2024

@author: ASUS
"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Data_analyses/mafic_and_gold_date_location_analyses.xlsx'
    data = pd.read_excel(file_path)
    types = ['Gabbro', 'Diabase', 'Lamprophyre', 'Orogenic']
    filtered_data = data[data['Type'].isin(types)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'hspace': 0})
    type_styles = {'Gabbro': {'color': (40 / 255, 181 / 255, 177 / 255), 'marker': 'o', 'label': 'Gabbro'}, 'Diabase': {'color': (51 / 255, 204 / 255, 0 / 255), 'marker': 'o', 'label': 'Diabase'}, 'Lamprophyre': {'color': (137 / 255, 137 / 255, 137 / 255), 'marker': 'o', 'label': 'Lamprophyre'}, 'Orogenic': {'color': (245 / 255, 196 / 255, 0 / 255), 'marker': 'o', 'label': 'Orogenic'}}
    for rock_type, style in type_styles.items():
        type_data = filtered_data[filtered_data['Type'] == rock_type]
        ax1.scatter(type_data['Mineralization period(Ma)'], [rock_type] * len(type_data), color=style['color'], marker=style['marker'], label=style['label'], s=100, edgecolor='black')
    ax1.set_ylabel('Rock Type', fontsize=12)
    ax1.legend(title='Rock Type', loc='upper right')
    ax1.invert_xaxis()
    highlight_points = [66, 251, 538.8]
    for point in highlight_points:
        ax1.axvline(point, color='red', linestyle='--', linewidth=1)
        ax1.text(point, 'Orogenic', f'{point}', color='red', ha='center', va='bottom', fontsize=10, fontweight='bold')
    file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Data_analyses/Date_Th_U/Data'
    file_1 = os.path.join(file_path, 'output_with_best_age_and_era_labels_Deleted.xlsx')
    file_2 = os.path.join(file_path, 'output_with_best_age_and_era_labels_Deleted_age_group_average.xlsx')
    df_file_1 = pd.read_excel(file_1)
    df_file_2 = pd.read_excel(file_2)
    df_file_1 = df_file_1[['Th/U', 'Best age']]
    df_file_2 = df_file_2[['Age Group', 'Th/U']]
    sns.lineplot(x='Age Group', y='Th/U', data=df_file_2, ax=ax2, color='red', label='Line plot', marker='o')
    sns.scatterplot(x='Best age', y='Th/U', data=df_file_1, ax=ax2, color='gray', label='Scatter plot', alpha=0.6, edgecolor=None)
    ax2.set_xlim(0, 1000)
    ax2_2 = ax2.twinx()
    sns.histplot(df_file_1['Best age'], bins=range(0, 1050, 50), color='lightblue', ax=ax2_2, label='Histogram', alpha=0.5)
    ax2_2.grid(False)
    ax2.set_xlabel('Best Age', fontsize=12)
    ax2.set_ylabel('Th/U', fontsize=12)
    ax2_2.set_ylabel('Frequency', fontsize=12)
    ax2.set_ylim(0, None)
    ax2.legend(loc='upper left')
    ax2_2.legend(loc='upper right')
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Data_analyses/Date_Th_U/Data'
    output_png = os.path.join(output_dir, 'Th_U_and_mafic_data.png')
    output_svg = os.path.join(output_dir, 'Th_U_and_mafic_data.svg')
    print(f'Output directory: {output_dir}')
    print(f'PNG file will be saved to: {output_png}')
    print(f'SVG file will be saved to: {output_svg}')
    plt.savefig(_output(output_png), format='png', bbox_inches='tight')
    plt.savefig(_output(output_svg), format='svg', bbox_inches='tight')
    plt.show()
    print(f'图形已保存为 PNG 和 SVG 格式：\n{output_png}\n{output_svg}')

def original_047():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from scipy.interpolate import CubicSpline
    import numpy as np
    file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Data_analyses/mafic_and_gold_date_location_analyses.xlsx'
    data = pd.read_excel(file_path)
    filtered_data = data[data['Type'] == 'Orogenic']
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True, gridspec_kw={'hspace': 0})
    ax1.scatter(filtered_data['Mineralization period(Ma)'], ['Orogenic'] * len(filtered_data), color=(245 / 255, 196 / 255, 0 / 255), marker='o', label='Orogenic', s=100, edgecolor='black')
    ax1.set_ylabel('Rock Type', fontsize=12)
    ax1.legend(title='Rock Type', loc='upper right')
    ax1.invert_xaxis()
    highlight_points = [66, 251, 538.8]
    for point in highlight_points:
        ax1.axvline(point, color='red', linestyle='--', linewidth=1)
        ax1.text(point, 'Orogenic', f'{point}', color='red', ha='center', va='bottom', fontsize=10, fontweight='bold')
    mineralization_period = filtered_data['Mineralization period(Ma)']
    n, bins, patches = ax2.hist(mineralization_period, bins=range(0, 1050, 50), color=(245 / 255, 196 / 255, 0 / 255), edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Mineralization Period (Ma)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Orogenic Gold Deposits Over Time', fontsize=14)
    ax2.grid(False)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    spline = CubicSpline(bin_centers, n)
    x_vals = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    y_vals = spline(x_vals)
    ax2.plot(x_vals, y_vals, color='black', label='Smoothed curve')
    ax2.legend(loc='upper right')
    file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Data_analyses/Date_Th_U/Data'
    file_1 = os.path.join(file_path, 'output_with_best_age_and_era_labels_Deleted.xlsx')
    file_2 = os.path.join(file_path, 'output_with_best_age_and_era_labels_Deleted_age_group_average.xlsx')
    df_file_1 = pd.read_excel(file_1)
    df_file_2 = pd.read_excel(file_2)
    df_file_1 = df_file_1[['Th/U', 'Best age']]
    df_file_2 = df_file_2[['Age Group', 'Th/U']]
    sns.lineplot(x='Age Group', y='Th/U', data=df_file_2, ax=ax3, color='red', label='Line plot', marker='o')
    sns.scatterplot(x='Best age', y='Th/U', data=df_file_1, ax=ax3, color='gray', label='Scatter plot', alpha=0.6, edgecolor=None)
    ax3.set_xlim(0, 1000)
    ax3_2 = ax3.twinx()
    n, bins, patches = ax3_2.hist(df_file_1['Best age'], bins=range(0, 1050, 50), color='lightblue', alpha=0.5, label='Histogram', edgecolor='black')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    spline = CubicSpline(bin_centers, n)
    x_vals = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    y_vals = spline(x_vals)
    ax3_2.plot(x_vals, y_vals, color='blue', label='Smoothed curve')
    ax3_2.legend(loc='upper right')
    ax3.set_xlabel('Best Age', fontsize=12)
    ax3.set_ylabel('Th/U', fontsize=12)
    ax3_2.set_ylabel('Frequency', fontsize=12)
    ax3.set_ylim(0, None)
    ax3.legend(loc='upper left')
    ax3_2.legend(loc='upper right')
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Data_analyses/Date_Th_U/Data'
    output_png = os.path.join(output_dir, 'Gold_deposit_and_detrital_zircon_Th_U_and_mafic_data.png')
    output_svg = os.path.join(output_dir, 'Gold_deposit_and_detrital_zircon_Th_U_and_mafic_data.svg')
    print(f'Output directory: {output_dir}')
    print(f'PNG file will be saved to: {output_png}')
    print(f'SVG file will be saved to: {output_svg}')
    plt.savefig(_output(output_png), format='png', bbox_inches='tight')
    plt.savefig(_output(output_svg), format='svg', bbox_inches='tight')
    plt.show()
    print(f'图形已保存为 PNG 和 SVG 格式：\n{output_png}\n{output_svg}')

def original_050():
    """
Created on Fri Jan 17 16:32:41 2025

@author: ASUS
"""
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Data_Analyses/mafic_and_gold_date_location_analyses.xlsx'
    data = pd.read_excel(file_path)
    data['Longitude'] = pd.to_numeric(data['Longitude'].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')
    data['Latitude'] = pd.to_numeric(data['Latitude'].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')
    data = data.dropna(subset=['Longitude', 'Latitude'])
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['Longitude'], data['Latitude']))
    rock_shapes = {'Gabbro': 'o', 'Diabase': 's', 'Lamprophyre': '^', 'Orogenic': '*'}
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(data['Mineralization period(Ma)'].min(), data['Mineralization period(Ma)'].max())
    fig, ax = plt.subplots(figsize=(21.4, 6.66), dpi=100)
    for rock_type, marker in rock_shapes.items():
        subset = data[data['Type'] == rock_type]
        sc = ax.scatter(subset['Longitude'], subset['Latitude'], s=300, c=subset['Mineralization period(Ma)'], cmap=cmap, norm=norm, marker=marker, label=rock_type, edgecolor='k')
    ax.set_xlim(55, 135)
    ax.set_ylim(35, 60)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Mineralization period (Ma)')
    ax.legend(loc='upper right')
    ax.set_title('Distribution of Mafic Rocks and Orogenic Gold with Age and Rock Types')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    output_png = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Data_Analyses/mafic_gold_distribution_custom_size_0117.png'
    output_svg = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Data_Analyses/mafic_gold_distribution_custom_size_0117.svg'
    plt.savefig(_output(output_png), format='png', dpi=300, bbox_inches='tight')
    plt.savefig(_output(output_svg), format='svg', bbox_inches='tight')
    plt.show()

def original_052():
    import pandas as pd
    import matplotlib.pyplot as plt
    file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Data_analyses/mafic_and_gold_date_location_analyses.xlsx'
    data = pd.read_excel(file_path)
    print(data.columns)
    types = ['Gabbro', 'Diabase', 'Lamprophyre', 'Orogenic']
    filtered_data = data[data['Type'].isin(types)]
    plt.figure(figsize=(12, 6))
    type_styles = {'Gabbro': {'color': (40 / 255, 181 / 255, 177 / 255), 'marker': 'o', 'label': 'Gabbro'}, 'Diabase': {'color': (51 / 255, 204 / 255, 0 / 255), 'marker': 'o', 'label': 'Diabase'}, 'Lamprophyre': {'color': (137 / 255, 137 / 255, 137 / 255), 'marker': 'o', 'label': 'Lamprophyre'}, 'Orogenic': {'color': (245 / 255, 196 / 255, 0 / 255), 'marker': 'o', 'label': 'Orogenic'}}
    for rock_type, style in type_styles.items():
        type_data = filtered_data[filtered_data['Type'] == rock_type]
        plt.scatter(type_data['Mineralization period(Ma)'], [rock_type] * len(type_data), color=style['color'], marker=style['marker'], label=style['label'], s=100, edgecolor='black')
    plt.legend(title='Rock Type')
    plt.xlabel('Mineralization Period (Ma)')
    plt.ylabel('Rock Type')
    plt.title('Central Asian Orogenic Belt Mineralization Periods by Rock Type')
    plt.gca().invert_xaxis()
    output_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Data_analyses'
    plt.savefig(_output(f'{output_path}/Central_Asian_Orogenic_Belt.png'), format='png', dpi=300)
    plt.savefig(_output(f'{output_path}/Central_Asian_Orogenic_Belt.svg'), format='svg')
    plt.show()

def original_054():
    """
Created on Thu Nov  7 15:26:24 2024

@author: ASUS
"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Flatten, Dense
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    import os
    train_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/grayscale_data_combined_train.csv'
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/grayscale_data_combined_test.csv'
    pca_data_train = pd.read_csv(train_file_path)
    pca_data_test = pd.read_csv(test_file_path)
    feature_names = pca_data_train.columns[-23:]
    train_pca_values = pca_data_train[feature_names].values
    train_labels = pca_data_train['Label'].values
    train_image = train_pca_values.reshape(-1, 23, 1)
    train_label_image = train_labels
    input_shape = (23, 1)
    input_combined = Input(shape=input_shape)
    x = Conv1D(64, 3, padding='same', activation='relu')(input_combined)
    x = BatchNormalization()(x)
    x = Conv1D(32, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    final_output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_combined, outputs=final_output)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/1DCNN'
    if not os.path.exists(output_dir):
        os.makedirs(str(OUT), exist_ok=True)
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-06)
    checkpoint_path = os.path.join(output_dir, 'best_model_origin_1DCNN.keras')
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='accuracy', save_best_only=True)
    history = model.fit(train_image, train_label_image, batch_size=32, epochs=100, verbose=2, callbacks=[early_stopping, reduce_lr, model_checkpoint])
    model_save_path = os.path.join(output_dir, 'trained_model_origin_1DCNN.h5')
    model.save(_output(model_save_path))
    print(f'训练好的模型已保存至: {model_save_path}')
    test_pca_values = pca_data_test[feature_names].values
    test_image = test_pca_values.reshape(-1, 23, 1)
    predictions = model.predict(test_image)
    pca_data_test['Predicted_Probability'] = predictions.flatten()
    output_file = os.path.join(output_dir, 'PCA_with_predictions_origin_1DCNN.csv')
    pca_data_test.to_csv(_output(output_file), index=False)
    print(f'完整预测结果已保存至 {output_file}')
    x_coords = pca_data_test['x'].values
    y_coords = pca_data_test['y'].values
    predicted_probs = pca_data_test['Predicted_Probability'].values
    plt.figure(figsize=(18, 5.64), dpi=100)
    plt.tricontourf(x_coords, y_coords, predicted_probs, levels=14, cmap='coolwarm')
    plt.colorbar(label='Predicted Probability')
    plt.axis('off')
    plt.title('Predicted Probability Contour Map 1DCNN')
    contour_png = os.path.join(output_dir, 'contour_map_origin_1DCNN.png')
    contour_svg = os.path.join(output_dir, 'contour_map_origin_1DCNN.svg')
    plt.savefig(_output(contour_png), dpi=100, bbox_inches='tight')
    plt.savefig(_output(contour_svg), dpi=100, bbox_inches='tight')
    plt.show()
    plt.style.use('default')
    plt.plot(history.history['accuracy'], label='Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.grid(False)
    accuracy_curve_png = os.path.join(output_dir, 'accuracy_curve_origin_1DCNN.png')
    accuracy_curve_svg = os.path.join(output_dir, 'accuracy_curve_origin_1DCNN.svg')
    plt.savefig(_output(accuracy_curve_png), dpi=100, bbox_inches='tight')
    plt.savefig(_output(accuracy_curve_svg), dpi=100, bbox_inches='tight')
    plt.show()
    plt.plot(history.history['loss'], label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(False)
    loss_curve_png = os.path.join(output_dir, 'loss_curve_origin_1DCNN.png')
    loss_curve_svg = os.path.join(output_dir, 'loss_curve_origin_1DCNN.svg')
    plt.savefig(_output(loss_curve_png), dpi=100, bbox_inches='tight')
    plt.savefig(_output(loss_curve_svg), dpi=100, bbox_inches='tight')
    plt.show()
    plt.scatter(range(len(history.history['accuracy'])), history.history['accuracy'], label='Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([min(history.history['accuracy']) - 0.05, max(history.history['accuracy']) + 0.05])
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.title('Training Accuracy Curve')
    accuracy_scatter_png = os.path.join(output_dir, 'accuracy_scatter_curve_origin_1DCNN.png')
    accuracy_scatter_svg = os.path.join(output_dir, 'accuracy_scatter_curve_origin_1DCNN.svg')
    plt.savefig(_output(accuracy_scatter_png))
    plt.savefig(_output(accuracy_scatter_svg), format='svg')
    plt.show()
    plt.scatter(range(len(history.history['loss'])), history.history['loss'], label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([min(history.history['loss']) - 0.05, max(history.history['loss']) + 0.05])
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.title('Training Loss Curve')
    loss_scatter_png = os.path.join(output_dir, 'loss_scatter_curve_origin_1DCNN.png')
    loss_scatter_svg = os.path.join(output_dir, 'loss_scatter_curve_origin_1DCNN.svg')
    plt.savefig(_output(loss_scatter_png))
    plt.savefig(_output(loss_scatter_svg), format='svg')
    plt.show()

    def pca_feature_sensitivity_analysis(model, test_patches, feature_index):
        modified_patches = np.copy(test_patches)
        modified_patches[:, feature_index, :] = 0
        original_predictions = model.predict(test_patches)
        modified_predictions = model.predict(modified_patches)
        predictions_diff = np.abs(original_predictions - modified_predictions)
        return predictions_diff
    feature_contributions = []
    for i in range(23):
        contribution = pca_feature_sensitivity_analysis(model, test_image, i)
        feature_contributions.append(np.mean(contribution))
    plt.figure(figsize=(10, 6))
    bars = plt.bar(feature_names, feature_contributions, color='skyblue', edgecolor='black', linewidth=1)
    plt.xlabel('PCA Feature')
    plt.ylabel('Average Contribution to Prediction')
    plt.title('PCA Feature Sensitivity Analysis')
    plt.xticks(rotation=90)
    feature_contributions_png = os.path.join(output_dir, 'pca_feature_contributions_origin_1DCNN.png')
    feature_contributions_svg = os.path.join(output_dir, 'pca_feature_contributions_origin_1DCNN.svg')
    plt.savefig(_output(feature_contributions_png), format='png', dpi=100, bbox_inches='tight')
    plt.savefig(_output(feature_contributions_svg), format='svg', bbox_inches='tight')
    plt.show()
    print(f'特征贡献度图已保存为: {feature_contributions_png} 和 {feature_contributions_svg}')

def original_055():
    """
Created on Thu Nov  7 10:39:31 2024

@author: ASUS
"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    import os
    train_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/grayscale_data_combined_train.csv'
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/grayscale_data_combined_test.csv'
    pca_data_train = pd.read_csv(train_file_path)
    pca_data_test = pd.read_csv(test_file_path)
    x_coords = pca_data_test['x'].values
    y_coords = pca_data_test['y'].values
    pca_values = pca_data_test.iloc[:, -23:].values

    def prepare_test_data_by_coordinates(x_coords, y_coords, pca_values, patch_size=9, min_points=50):
        reshaped_patches = []
        indices = []
        min_x, max_x = (min(x_coords), max(x_coords))
        min_y, max_y = (min(y_coords), max(y_coords))
        x_steps = np.arange(min_x, max_x + 1, patch_size)
        y_steps = np.arange(min_y, max_y + 1, patch_size)
        for y_min in y_steps:
            y_max = y_min + patch_size - 1
            if y_max > max_y:
                continue
            for x_min in x_steps:
                x_max = x_min + patch_size - 1
                if x_max > max_x:
                    continue
                mask = (x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)
                patch_data = pca_values[mask]
                num_points = len(patch_data)
                if num_points >= min_points:
                    if num_points < patch_size * patch_size:
                        patch_data = np.pad(patch_data, ((0, patch_size * patch_size - num_points), (0, 0)), mode='constant')
                    try:
                        reshaped_patch = patch_data.reshape(patch_size, patch_size, -1)
                        reshaped_patches.append(reshaped_patch)
                        indices.append((x_min, x_max, y_min, y_max))
                    except ValueError as e:
                        print(f'Error reshaping patch at X: {x_min}-{x_max}, Y: {y_min}-{y_max} -> {e}')
                else:
                    print(f'Skipping patch at X: {x_min}-{x_max}, Y: {y_min}-{y_max}, insufficient points.')
        print(f'Total patches generated: {len(reshaped_patches)}')
        return (np.array(reshaped_patches), indices)
    input_shape = (9, 9, 23)
    input_combined = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_combined)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    final_output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_combined, outputs=final_output)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses'
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=1e-06)
    checkpoint_path = os.path.join(output_dir, 'best_model_9x9_patches_2DCNN_original_analyse.keras')
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='loss', save_best_only=True)
    positive_samples = pca_data_train[pca_data_train['Label'] == 1]
    negative_samples = pca_data_train[pca_data_train['Label'] == 0]

    def extract_and_reshape_samples(samples, patch_size=9):
        reshaped_patches = []
        for i in range(0, len(samples), patch_size * patch_size):
            patch = samples.iloc[i:i + patch_size * patch_size]
            if len(patch) == patch_size * patch_size:
                reshaped_patch = patch.iloc[:, -23:].values.reshape(patch_size, patch_size, -1)
                reshaped_patches.append(reshaped_patch)
        return np.array(reshaped_patches)
    positive_patches = extract_and_reshape_samples(positive_samples)
    negative_patches = extract_and_reshape_samples(negative_samples)
    combined_patches = np.concatenate((positive_patches, negative_patches), axis=0)
    labels = np.concatenate((np.ones(len(positive_patches)), np.zeros(len(negative_patches))))
    history = model.fit(combined_patches, labels, batch_size=32, epochs=100, verbose=2, callbacks=[early_stopping, reduce_lr, model_checkpoint])
    model_save_path = os.path.join(output_dir, '2DCNN_model_original_analyse.h5')
    model.save(_output(model_save_path))
    plt.scatter(range(len(history.history['accuracy'])), history.history['accuracy'], color='blue', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([min(history.history['accuracy']) - 0.05, max(history.history['accuracy']) + 0.05])
    plt.legend(loc='lower right')
    plt.title('Training Accuracy Curve (Scatter Plot)')
    plt.savefig(_output(os.path.join(output_dir, 'accuracy_curve_original_analyse.png')))
    plt.savefig(_output(os.path.join(output_dir, 'accuracy_curve_original_analyse.svg')), format='svg')
    plt.show()
    plt.scatter(range(len(history.history['loss'])), history.history['loss'], color='blue', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training Loss Curve (Scatter Plot)')
    plt.savefig(_output(os.path.join(output_dir, 'loss_curve_original_analyse.png')))
    plt.savefig(_output(os.path.join(output_dir, 'loss_curve_original_analyse.svg')), format='svg')
    plt.show()
    test_patches, indices = prepare_test_data_by_coordinates(x_coords, y_coords, pca_values, patch_size=9, min_points=50)
    predictions = model.predict(test_patches)
    pca_data_test['Predicted_Probability'] = np.nan
    for idx, (x_min, x_max, y_min, y_max) in enumerate(indices):
        mask = (x_coords >= x_min) & (x_coords < x_max) & (y_coords >= y_min) & (y_coords < y_max)
        pca_data_test.loc[mask, 'Predicted_Probability'] = np.full(mask.sum(), predictions[idx])
    output_file = os.path.join(output_dir, 'original_analyse_with_full_predictions_Test_2DCNN.csv')
    pca_data_test.to_csv(_output(output_file), index=False)
    valid_indices = ~np.isnan(pca_data_test['Predicted_Probability'].values)
    x_coords, y_coords, predicted_probs = (x_coords[valid_indices], y_coords[valid_indices], pca_data_test['Predicted_Probability'].values[valid_indices])
    plt.figure(figsize=(18, 5.64), dpi=100)
    plt.tricontourf(x_coords, y_coords, predicted_probs, levels=14, cmap='coolwarm')
    plt.colorbar(label='Predicted Probability')
    plt.title('Predicted Probability Contour Map 2DCNN')
    plt.axis('off')
    plt.savefig(_output(os.path.join(output_dir, 'contour_map_original_analyse.png')), dpi=100, bbox_inches='tight')
    plt.savefig(_output(os.path.join(output_dir, 'contour_map_original_analyse.svg')), format='svg')
    plt.show()
    feature_names = pca_data_train.columns[-23:]

    def pca_feature_sensitivity_analysis(model, test_patches, feature_index):
        modified_patches = np.copy(test_patches)
        modified_patches[:, :, :, feature_index] = 0
        original_predictions = model.predict(test_patches)
        modified_predictions = model.predict(modified_patches)
        predictions_diff = np.abs(original_predictions - modified_predictions)
        return predictions_diff
    feature_contributions = [np.mean(pca_feature_sensitivity_analysis(model, test_patches, i)) for i in range(23)]
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, feature_contributions, color='skyblue', edgecolor='black')
    plt.xlabel('Feature')
    plt.ylabel('Average Contribution to Prediction')
    plt.title('Feature Sensitivity Analysis')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(_output(os.path.join(output_dir, 'feature_contributions_original_analyse.png')), format='png', dpi=100, bbox_inches='tight')
    plt.savefig(_output(os.path.join(output_dir, 'feature_contributions_original_analyse.svg')), format='svg', bbox_inches='tight')
    plt.show()

def original_056():
    """
Created on Mon Dec 23 21:27:50 2024

@author: ASUS
"""
    "\nCreated on Thu Nov 7 2024\nRe-using the trained 2DCNN model to run the same analysis and save outputs with 'origin_2DCNN_frozen' prefix.\n"
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    import os
    model_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/2DCNN_model_original_analyse.h5'
    model = load_model(model_path)
    train_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/grayscale_data_combined_train.csv'
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/grayscale_data_combined_test.csv'
    pca_data_train = pd.read_csv(train_file_path)
    pca_data_test = pd.read_csv(test_file_path)
    x_coords = pca_data_test['x'].values
    y_coords = pca_data_test['y'].values
    pca_values = pca_data_test.iloc[:, -23:].values

    def prepare_test_data_by_coordinates(x_coords, y_coords, pca_values, patch_size=9, min_points=50):
        reshaped_patches = []
        indices = []
        min_x, max_x = (min(x_coords), max(x_coords))
        min_y, max_y = (min(y_coords), max(y_coords))
        x_steps = np.arange(min_x, max_x + 1, patch_size)
        y_steps = np.arange(min_y, max_y + 1, patch_size)
        for y_min in y_steps:
            y_max = y_min + patch_size - 1
            if y_max > max_y:
                continue
            for x_min in x_steps:
                x_max = x_min + patch_size - 1
                if x_max > max_x:
                    continue
                mask = (x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)
                patch_data = pca_values[mask]
                num_points = len(patch_data)
                if num_points >= min_points:
                    if num_points < patch_size * patch_size:
                        patch_data = np.pad(patch_data, ((0, patch_size * patch_size - num_points), (0, 0)), mode='constant')
                    try:
                        reshaped_patch = patch_data.reshape(patch_size, patch_size, -1)
                        reshaped_patches.append(reshaped_patch)
                        indices.append((x_min, x_max, y_min, y_max))
                    except ValueError as e:
                        print(f'Error reshaping patch at X: {x_min}-{x_max}, Y: {y_min}-{y_max} -> {e}')
                else:
                    print(f'Skipping patch at X: {x_min}-{x_max}, Y: {y_min}-{y_max}, insufficient points.')
        print(f'Total patches generated: {len(reshaped_patches)}')
        return (np.array(reshaped_patches), indices)
    test_patches, indices = prepare_test_data_by_coordinates(x_coords, y_coords, pca_values, patch_size=9, min_points=50)
    predictions = model.predict(test_patches)
    pca_data_test['Predicted_Probability'] = np.nan
    for idx, (x_min, x_max, y_min, y_max) in enumerate(indices):
        mask = (x_coords >= x_min) & (x_coords < x_max) & (y_coords >= y_min) & (y_coords < y_max)
        pca_data_test.loc[mask, 'Predicted_Probability'] = np.full(mask.sum(), predictions[idx])
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/2DCNN/frozen_test'
    output_file = os.path.join(output_dir, 'origin_2DCNN_frozen_predictions_Test_2DCNN.csv')
    pca_data_test.to_csv(_output(output_file), index=False)
    plt.scatter(range(len(predictions)), predictions, color='blue', label='Predicted Probability')
    plt.xlabel('Index')
    plt.ylabel('Predicted Probability')
    plt.legend(loc='lower right')
    plt.title('Prediction Scatter Plot (Frozen Model)')
    plt.savefig(_output(os.path.join(output_dir, 'origin_2DCNN_frozen_accuracy_curve.png')))
    plt.savefig(_output(os.path.join(output_dir, 'origin_2DCNN_frozen_accuracy_curve.svg')), format='svg')
    plt.show()
    plt.scatter(range(len(predictions)), np.log(1 + np.abs(predictions - 0.5)), color='blue', label='Loss')
    plt.xlabel('Index')
    plt.ylabel('Loss (Log Transformed)')
    plt.legend(loc='upper right')
    plt.title('Prediction Loss Curve (Frozen Model)')
    plt.savefig(_output(os.path.join(output_dir, 'origin_2DCNN_frozen_loss_curve.png')))
    plt.savefig(_output(os.path.join(output_dir, 'origin_2DCNN_frozen_loss_curve.svg')), format='svg')
    plt.show()
    valid_indices = ~np.isnan(pca_data_test['Predicted_Probability'].values)
    x_coords, y_coords, predicted_probs = (x_coords[valid_indices], y_coords[valid_indices], pca_data_test['Predicted_Probability'].values[valid_indices])
    plt.figure(figsize=(18, 5.64), dpi=100)
    plt.tricontourf(x_coords, y_coords, predicted_probs, levels=14, cmap='coolwarm')
    plt.colorbar(label='Predicted Probability')
    plt.title('Predicted Probability Contour Map (Frozen Model)')
    plt.axis('off')
    plt.savefig(_output(os.path.join(output_dir, 'origin_2DCNN_frozen_contour_map.png')), dpi=100, bbox_inches='tight')
    plt.savefig(_output(os.path.join(output_dir, 'origin_2DCNN_frozen_contour_map.svg')), format='svg')
    plt.show()

    def pca_feature_sensitivity_analysis(model, test_patches, feature_index):
        modified_patches = np.copy(test_patches)
        modified_patches[:, :, :, feature_index] = 0
        original_predictions = model.predict(test_patches)
        modified_predictions = model.predict(modified_patches)
        predictions_diff = np.abs(original_predictions - modified_predictions)
        return predictions_diff
    feature_names = pca_data_train.columns[-23:]
    feature_contributions = [np.mean(pca_feature_sensitivity_analysis(model, test_patches, i)) for i in range(23)]
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, feature_contributions, color='skyblue', edgecolor='black')
    plt.xlabel('Feature')
    plt.ylabel('Average Contribution to Prediction')
    plt.title('Feature Sensitivity Analysis (Frozen Model)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(_output(os.path.join(output_dir, 'origin_2DCNN_frozen_feature_contributions.png')), format='png', dpi=100, bbox_inches='tight')
    plt.savefig(_output(os.path.join(output_dir, 'origin_2DCNN_frozen_feature_contributions.svg')), format='svg', bbox_inches='tight')
    plt.show()

def original_060():
    """
Created on Sat Sep 28 14:56:25 2024

@author: ASUS
"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    train_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/PCA_with_labels_Train.csv'
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/PCA_with_labels_Test.csv'
    pca_data_train = pd.read_csv(train_file_path)
    pca_data_test = pd.read_csv(test_file_path)
    x_coords = pca_data_test['x'].values
    y_coords = pca_data_test['y'].values
    pca_values = pca_data_test.iloc[:, -23:].values

    def prepare_test_data_by_coordinates(x_coords, y_coords, pca_values, patch_size=9, min_points=50):
        """
    按步长划分测试集为 patch_size*patch_size 的块，X 和 Y 坐标按 9 递增。
    Args:
    - x_coords: X 坐标数组
    - y_coords: Y 坐标数组
    - pca_values: PCA 特征数组
    - patch_size: 块的大小，默认为 9
    - min_points: 每个块的最小点数，默认为 50
    """
        reshaped_patches = []
        indices = []
        min_x, max_x = (min(x_coords), max(x_coords))
        min_y, max_y = (min(y_coords), max(y_coords))
        print(f'X range: {min_x} to {max_x}')
        print(f'Y range: {min_y} to {max_y}')
        x_steps = np.arange(min_x, max_x + 1, patch_size)
        y_steps = np.arange(min_y, max_y + 1, patch_size)
        for y_min in y_steps:
            y_max = y_min + patch_size - 1
            if y_max > max_y:
                continue
            for x_min in x_steps:
                x_max = x_min + patch_size - 1
                if x_max > max_x:
                    continue
                mask = (x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)
                patch_data = pca_values[mask]
                num_points = len(patch_data)
                if num_points >= min_points:
                    if num_points < patch_size * patch_size:
                        patch_data = np.pad(patch_data, ((0, patch_size * patch_size - num_points), (0, 0)), mode='constant')
                    try:
                        reshaped_patch = patch_data.reshape(patch_size, patch_size, -1)
                        reshaped_patches.append(reshaped_patch)
                        indices.append((x_min, x_max, y_min, y_max))
                    except ValueError as e:
                        print(f'Error reshaping patch at X: {x_min}-{x_max}, Y: {y_min}-{y_max} -> {e}')
                else:
                    print(f'Skipping patch at X: {x_min}-{x_max}, Y: {y_min}-{y_max}, insufficient points.')
        print(f'Total patches generated: {len(reshaped_patches)}')
        return (np.array(reshaped_patches), indices)
    input_shape = (9, 9, 23)
    input_combined = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_combined)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    final_output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_combined, outputs=final_output)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=1e-06)
    checkpoint_path = 'best_model_9x9_patches_2DCNN.keras'
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='loss', save_best_only=True)
    positive_samples = pca_data_train[pca_data_train['Label'] == 1]
    negative_samples = pca_data_train[pca_data_train['Label'] == 0]

    def extract_and_reshape_samples(samples, patch_size=9):
        """从样本中提取 9x9 块并将其拼接成矩阵"""
        reshaped_patches = []
        for i in range(0, len(samples), patch_size * patch_size):
            patch = samples.iloc[i:i + patch_size * patch_size]
            if len(patch) == patch_size * patch_size:
                reshaped_patch = patch.iloc[:, -23:].values.reshape(patch_size, patch_size, -1)
                reshaped_patches.append(reshaped_patch)
        return np.array(reshaped_patches)
    positive_patches = extract_and_reshape_samples(positive_samples)
    negative_patches = extract_and_reshape_samples(negative_samples)
    combined_patches = np.concatenate((positive_patches, negative_patches), axis=0)
    labels = np.concatenate((np.ones(len(positive_patches)), np.zeros(len(negative_patches))))
    history = model.fit(combined_patches, labels, batch_size=32, epochs=100, verbose=2, callbacks=[early_stopping, reduce_lr, model_checkpoint])
    plt.plot(history.history['accuracy'], label='Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.title('Training Accuracy Curve')
    plt.savefig(_output('accuracy_curve_2DCNN.png'))
    plt.savefig(_output('accuracy_curve_2DCNN.svg'), format='svg')
    plt.show()
    plt.plot(history.history['loss'], label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.title('Training Loss Curve')
    plt.savefig(_output('loss_curve_2DCNN.png'))
    plt.savefig(_output('loss_curve_2DCNN.svg'), format='svg')
    plt.show()
    test_patches, indices = prepare_test_data_by_coordinates(x_coords, y_coords, pca_values, patch_size=9, min_points=50)
    print(f'Test patches shape: {test_patches.shape}')
    if test_patches.shape[1:] == (9, 9, 23):
        predictions = model.predict(test_patches)
    else:
        raise ValueError(f'Invalid shape for test_patches: {test_patches.shape}, expected (num_samples, 9, 9, 23)')
    pca_data_test['Predicted_Probability'] = np.nan
    for idx, (x_min, x_max, y_min, y_max) in enumerate(indices):
        mask = (x_coords >= x_min) & (x_coords < x_max) & (y_coords >= y_min) & (y_coords < y_max)
        pca_data_test.loc[mask, 'Predicted_Probability'] = np.full(mask.sum(), predictions[idx])
    output_file = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/PCA_with_full_predictions_Test_2DCNN.csv'
    pca_data_test.to_csv(_output(output_file), index=False)
    print(f'完整预测结果已保存至 {output_file}')
    x_coords = pca_data_test['x'].values
    y_coords = pca_data_test['y'].values
    predicted_probs = pca_data_test['Predicted_Probability'].values
    valid_indices = ~np.isnan(predicted_probs)
    x_coords = x_coords[valid_indices]
    y_coords = y_coords[valid_indices]
    predicted_probs = predicted_probs[valid_indices]
    plt.figure(figsize=(18, 5.64), dpi=100)
    plt.tricontourf(x_coords, y_coords, predicted_probs, levels=14, cmap='coolwarm')
    plt.colorbar(label='Predicted Probability')
    plt.axis('off')
    plt.title('Predicted Probability Contour Map 2DCNN')
    plt.savefig(_output('contour_map_2DCNN.png'), dpi=100, bbox_inches='tight')
    plt.savefig(_output('contour_map_2DCNN.svg'), format='svg', bbox_inches='tight')
    plt.show()
    plt.scatter(range(len(history.history['accuracy'])), history.history['accuracy'], label='Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([min(history.history['accuracy']) - 0.05, max(history.history['accuracy']) + 0.05])
    plt.yticks([round(i * 0.1, 2) for i in range(7, 11)])
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.title('Training Accuracy Curve')
    plt.savefig(_output('accuracy_curve_2DCNN_scatter_v3.png'))
    plt.savefig(_output('accuracy_curve_2DCNN_scatter_v3.svg'), format='svg')
    plt.show()
    plt.scatter(range(len(history.history['loss'])), history.history['loss'], label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([min(history.history['loss']) - 0.05, max(history.history['loss']) + 0.05])
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.title('Training Loss Curve')
    plt.savefig(_output('loss_curve_2DCNN_scatter.png'))
    plt.savefig(_output('loss_curve_2DCNN_scatter.svg'), format='svg')
    plt.show()

def original_066():
    """
Created on Mon Dec 16 10:59:56 2024

@author: ASUS
"""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from matplotlib import font_manager
    import os
    input_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA_test/Mafic rock_only/mafic_only_PCA_test.csv'
    df = pd.read_csv(input_path)
    df_pca = df.loc[:, 'Al2O3':'TiO2']
    df_tag = df.loc[:, ['X', 'Y']]
    df_pca_processed = (df_pca - df_pca.mean()) / df_pca.std()
    cov_matrix = df_pca_processed.cov()
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio) * 100
    output_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA_test/Mafic rock_only'
    if not os.path.exists(output_path):
        os.makedirs(str(OUT), exist_ok=True)
    fig, ax1 = plt.subplots()
    ax1.bar(range(1, len(eigenvalues) + 1), eigenvalues, color='steelblue', alpha=0.7)
    ax1.set_xlabel('主成分', fontsize=12)
    ax1.set_ylabel('特征值', fontsize=12)
    ax1.set_ylim(0, max(eigenvalues) * 1.1)
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', color='red')
    ax2.set_ylabel('累计方差贡献率 (%)', fontsize=12)
    ax2.set_ylim(0, 110)
    ax1.set_xticks(range(1, len(eigenvalues) + 1))
    ax1.set_xticklabels([f'PCA{i}' for i in range(1, len(eigenvalues) + 1)])
    plt.title('主成分分析的特征值与累计方差贡献率', fontsize=14)
    plt.savefig(_output(f'{output_path}\\mafic_only_碎石图_with_cumulative_variance.png'))
    plt.savefig(_output(f'{output_path}\\mafic_only_碎石图_with_cumulative_variance.svg'))
    plt.show()
    n_components = 11
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_pca_processed)
    df_pca_result = pd.DataFrame(data=pca_result, columns=[f'PCA{i}' for i in range(1, n_components + 1)])
    df_merge = pd.merge(df_tag, df_pca_result, left_index=True, right_index=True, how='outer')
    df_merge.to_csv(_output(f'{output_path}\\mafic_only_PCA_result.csv'), index=False)
    yahei_font_path = 'C:\\Windows\\Fonts\\calibrib.ttf'
    custom_font = font_manager.FontProperties(fname=yahei_font_path)
    for i in range(n_components):
        plt.figure(figsize=(6, 10))
        pca_component = pca.components_[i]
        sorted_indices = pca_component.argsort()
        sorted_component = pca_component[sorted_indices]
        sorted_features = df_pca.columns[sorted_indices]
        colors = ['blue' if val > 0 else 'red' for val in sorted_component]
        plt.barh(sorted_features, sorted_component, color=colors, edgecolor='black')
        plt.axvline(x=0, color='black', linewidth=1, linestyle='-')
        plt.yticks(fontsize=14, fontproperties=custom_font)
        plt.xticks(fontsize=14, fontproperties=custom_font)
        plt.ylabel('Features', fontsize=24, fontweight='bold', fontproperties=custom_font)
        plt.xlabel('Contribution score', fontsize=24, fontweight='bold', fontproperties=custom_font)
        plt.title(f'PCA{i + 1} feature contributions', fontsize=24, fontweight='bold', fontproperties=custom_font)
        plt.tight_layout()
        plt.savefig(_output(f'{output_path}\\mafic_only_PCA{i + 1}_贡献度柱状图_sorted_horizontal.png'))
        plt.savefig(_output(f'{output_path}\\mafic_only_PCA{i + 1}_贡献度柱状图_sorted_horizontal.svg'))
        plt.show()

def original_068():
    """
Created on Tue Oct 22 16:59:45 2024

@author: ASUS
"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    import os
    pca_data = REFIT_SCORES.copy()
    x = pca_data['X']
    y = pca_data['Y']
    pca_columns = [f'PCA{i}' for i in range(1, 24)]
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA_test'
    os.makedirs(str(OUT), exist_ok=True)
    grid_x, grid_y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    for pca in pca_columns:
        z = pca_data[pca]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
        plt.figure(figsize=(18, 5.64), dpi=100)
        contour = plt.contourf(grid_x, grid_y, grid_z, levels=14, cmap='coolwarm')
        plt.colorbar(contour)
        plt.title(f'{pca} 等值线图')
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])
        file_name_base = os.path.join(output_dir, pca)
        plt.savefig(_output(f'{file_name_base}.png'), format='png')
        plt.savefig(_output(f'{file_name_base}.svg'), format='svg')
        plt.close()

def original_069():
    import numpy as np
    import os
    import pandas as pd
    from PIL import Image
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import seaborn as sns
    from matplotlib import font_manager

    def load_and_normalize_images(directory, target_size):
        image_list = []
        filenames = []
        if not os.path.exists(directory):
            raise FileNotFoundError(f'目录 {directory} 不存在。')
        print('开始加载图片...')
        for i, filename in enumerate(os.listdir(directory)):
            if filename.endswith('.jpg'):
                image_path = os.path.join(directory, filename)
                try:
                    image = Image.open(image_path).convert('RGB')
                    resized_image = image.resize(target_size)
                    normalized_image = np.array(resized_image) / 255.0
                    image_list.append(normalized_image)
                    filenames.append(filename)
                    print(f'已加载图片 {i + 1}/{len(os.listdir(directory))}: {filename}')
                except Exception as e:
                    print(f'加载图片 {filename} 时发生错误: {e}')
        print('图片加载完成！')
        return (np.array(image_list), filenames)

    def rgb_to_grayscale(image):
        return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

    def grayscale_images_to_dataframe(images, filenames):
        height, width = images[0].shape[:2]
        data = {'X': [], 'Y': []}
        print('开始转换为灰度图...')
        grayscale_images = []
        for i, image in enumerate(images):
            grayscale_image = rgb_to_grayscale(image)
            grayscale_images.append(grayscale_image)
            print(f'已转换 {i + 1}/{len(images)} 张图片为灰度图: {filenames[i]}')
        print('灰度图转换完成！')
        for filename in filenames:
            data[f'{filename}_Grayscale'] = []
        print('开始处理每个像素的灰度值...')
        for y in range(height):
            for x in range(width):
                data['X'].append(x)
                data['Y'].append(y)
                for i, grayscale_image in enumerate(grayscale_images):
                    data[f'{filenames[i]}_Grayscale'].append(grayscale_image[y, x])
            if y % 50 == 0:
                print(f'已处理 {y}/{height} 行的像素数据...')
        print('像素数据处理完成！')
        return pd.DataFrame(data)

    def export_grayscale_to_csv(images, filenames, output_file):
        print('开始将灰度数据导出到 CSV 文件...')
        df = grayscale_images_to_dataframe(images, filenames)
        df.to_csv(_output(output_file), index=False)
        print(f'灰度数据已成功导出到 {output_file}')
    data_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Figure'
    target_size = (1800, 564)
    output_file = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA_test/PCA_test.csv'
    images, filenames = load_and_normalize_images(data_dir, target_size)
    export_grayscale_to_csv(images, filenames, output_file)
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    df = pd.read_csv('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA/grayscale_data_combined.csv')
    output_fodel = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA_test/PCA_test'
    df_pca = df.loc[:, '176Hf-177Hf':'Hf(t)']
    df_tag = df.loc[:, ['X', 'Y']]
    df_pca_processed = (df_pca - df_pca.mean()) / df_pca.std()
    cov_matrix = df_pca_processed.cov()
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio) * 100
    fig, ax1 = plt.subplots()
    ax1.bar(range(1, len(eigenvalues) + 1), eigenvalues, color='steelblue', alpha=0.7)
    ax1.set_xlabel('主成分', fontsize=12)
    ax1.set_ylabel('特征值', fontsize=12)
    ax1.set_ylim(0, max(eigenvalues) * 1.1)
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', color='red')
    ax2.set_ylabel('累计方差贡献率 (%)', fontsize=12)
    ax2.set_ylim(0, 110)
    ax1.set_xticks(range(1, len(eigenvalues) + 1))
    ax1.set_xticklabels([f'PCA{i}' for i in range(1, len(eigenvalues) + 1)])
    plt.title('主成分分析的特征值与累计方差贡献率', fontsize=14)
    scree_plot_path_png = os.path.join(output_fodel, '碎石图_with_cumulative_variance.png')
    scree_plot_path_svg = os.path.join(output_fodel, '碎石图_with_cumulative_variance.svg')
    plt.savefig(_output(scree_plot_path_png))
    plt.savefig(_output(scree_plot_path_svg))
    plt.show()
    pca_variance_df = pd.DataFrame({'Principal Component': [f'PCA{i + 1}' for i in range(len(eigenvalues))], 'Eigenvalues': eigenvalues, 'Explained Variance Ratio': explained_variance_ratio, 'Cumulative Variance (%)': cumulative_variance})
    variance_output_path = os.path.join(output_fodel, 'PCA_variance_contributions.xlsx')
    pca_variance_df.to_excel(_output(variance_output_path), index=False)
    print(f'PCA 特征值和累计方差贡献率已导出到 {variance_output_path}')
    n_components = 23
    pca = PCA(n_components=n_components)
    pca.fit(df_pca_processed)
    pca_result = pca.transform(df_pca_processed)
    df_pca_result = pd.DataFrame(data=pca_result, columns=[f'PCA{i}' for i in range(1, n_components + 1)])
    df_merge = pd.merge(df_tag, df_pca_result, left_index=True, right_index=True, how='outer')
    df_merge.to_csv(_output('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA_test/PCA_test/PCA_03_01.csv'), index=False)
    yahei_font_path = 'C:\\Windows\\Fonts\\calibrib.ttf'
    custom_font = font_manager.FontProperties(fname=yahei_font_path)
    for i in range(n_components):
        plt.figure(figsize=(6, 10))
        pca_component = pca.components_[i]
        sorted_indices = pca_component.argsort()
        sorted_component = pca_component[sorted_indices]
        sorted_features = df_pca.columns[sorted_indices]
        colors = ['blue' if val > 0 else 'red' for val in sorted_component]
        plt.barh(sorted_features, sorted_component, color=colors, edgecolor='black')
        plt.axvline(x=0, color='black', linewidth=1, linestyle='-')
        plt.yticks(fontsize=14, fontproperties=custom_font)
        plt.xticks(fontsize=14, fontproperties=custom_font)
        plt.ylabel('Features', fontsize=24, fontweight='bold', fontproperties=custom_font)
        plt.xlabel('Contribution score', fontsize=24, fontweight='bold', fontproperties=custom_font)
        plt.title(f'PCA{i + 1} feature contributions', fontsize=24, fontweight='bold', fontproperties=custom_font)
        plt.tight_layout()
        pca_contribution_png = os.path.join(output_fodel, f'PCA{i + 1}_贡献度柱状图_sorted_horizontal.png')
        pca_contribution_svg = os.path.join(output_fodel, f'PCA{i + 1}_贡献度柱状图_sorted_horizontal.svg')
        plt.savefig(_output(pca_contribution_png))
        plt.savefig(_output(pca_contribution_svg))
        plt.show()

def original_070():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from matplotlib import font_manager
    import os
    input_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA_test/Zircon_only_PCA/Zircon_only_test.csv'
    df = pd.read_csv(input_path)
    df_pca = df.loc[:, '176Hf-177Hf':'Hf(t)']
    df_tag = df.loc[:, ['x', 'y', 'Label']]
    df_pca_processed = (df_pca - df_pca.mean()) / df_pca.std()
    cov_matrix = df_pca_processed.cov()
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio) * 100
    output_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA_test/Zircon_only_PCA'
    if not os.path.exists(output_path):
        os.makedirs(str(OUT), exist_ok=True)
    fig, ax1 = plt.subplots()
    ax1.bar(range(1, len(eigenvalues) + 1), eigenvalues, color='steelblue', alpha=0.7)
    ax1.set_xlabel('主成分', fontsize=12)
    ax1.set_ylabel('特征值', fontsize=12)
    ax1.set_ylim(0, max(eigenvalues) * 1.1)
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', color='red')
    ax2.set_ylabel('累计方差贡献率 (%)', fontsize=12)
    ax2.set_ylim(0, 110)
    ax1.set_xticks(range(1, len(eigenvalues) + 1))
    ax1.set_xticklabels([f'PCA{i}' for i in range(1, len(eigenvalues) + 1)])
    plt.title('主成分分析的特征值与累计方差贡献率', fontsize=14)
    plt.savefig(_output(f'{output_path}\\碎石图_with_cumulative_variance_Zircon_only.png'))
    plt.savefig(_output(f'{output_path}\\碎石图_with_cumulative_variance_Zircon_only.svg'))
    plt.show()
    n_components = 12
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_pca_processed)
    df_pca_result = pd.DataFrame(data=pca_result, columns=[f'PCA{i}' for i in range(1, n_components + 1)])
    df_merge = pd.merge(df_tag, df_pca_result, left_index=True, right_index=True, how='outer')
    df_merge.to_csv(_output(f'{output_path}\\PCA_Zircon_only.csv'), index=False)
    yahei_font_path = 'C:\\Windows\\Fonts\\calibrib.ttf'
    custom_font = font_manager.FontProperties(fname=yahei_font_path)
    for i in range(n_components):
        plt.figure(figsize=(6, 10))
        pca_component = pca.components_[i]
        sorted_indices = pca_component.argsort()
        sorted_component = pca_component[sorted_indices]
        sorted_features = df_pca.columns[sorted_indices]
        colors = ['blue' if val > 0 else 'red' for val in sorted_component]
        plt.barh(sorted_features, sorted_component, color=colors, edgecolor='black')
        plt.axvline(x=0, color='black', linewidth=1, linestyle='-')
        plt.yticks(fontsize=14, fontproperties=custom_font)
        plt.xticks(fontsize=14, fontproperties=custom_font)
        plt.ylabel('Features', fontsize=24, fontweight='bold', fontproperties=custom_font)
        plt.xlabel('Contribution score', fontsize=24, fontweight='bold', fontproperties=custom_font)
        plt.title(f'PCA{i + 1} feature contributions', fontsize=24, fontweight='bold', fontproperties=custom_font)
        plt.tight_layout()
        plt.savefig(_output(f'{output_path}\\PCA{i + 1}_贡献度柱状图_sorted_horizontal_Zircon_only.png'))
        plt.savefig(_output(f'{output_path}\\PCA{i + 1}_贡献度柱状图_sorted_horizontal_Zircon_only.svg'))
        plt.show()

def original_071():
    """
Created on Tue Nov 19 19:50:38 2024

@author: ASUS
"""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from matplotlib import font_manager
    import os
    input_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA_test/Zircon_only_PCA/监督-PLS-DA/Zircon_only_test_PLS-DA.csv'
    df = pd.read_csv(input_path)
    X = df.loc[:, '176Hf-177Hf':'Hf(t)']
    y = df['Label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    output_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/PCA_test/Zircon_only_PCA/监督-PLS-DA'
    if not os.path.exists(output_path):
        os.makedirs(str(OUT), exist_ok=True)
    n_components = 2
    pls = PLSRegression(n_components=n_components)
    X_pls = pls.fit_transform(X_scaled, y)[0]
    df_pls_result = pd.DataFrame(X_pls, columns=[f'PLS{i + 1}' for i in range(n_components)])
    df_pls_result['Label'] = y
    df_pls_result.to_csv(_output(f'{output_path}\\PLS-DA_Result.csv'), index=False)
    plt.figure(figsize=(8, 6))
    colors = ['blue' if label == 0 else 'red' for label in y]
    plt.scatter(df_pls_result['PLS1'], df_pls_result['PLS2'], c=colors, alpha=0.7, edgecolor='k')
    plt.title('PLS-DA Projection', fontsize=16)
    plt.xlabel('PLS1', fontsize=14)
    plt.ylabel('PLS2', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_output(f'{output_path}\\PLS-DA_Projection.png'))
    plt.savefig(_output(f'{output_path}\\PLS-DA_Projection.svg'))
    plt.show()
    pls_weights = pd.DataFrame(pls.x_weights_, index=X.columns, columns=[f'PLS{i + 1}_weight' for i in range(n_components)])
    pls_weights.to_csv(_output(f'{output_path}\\PLS-DA_Weights.csv'))
    yahei_font_path = 'C:\\Windows\\Fonts\\calibrib.ttf'
    custom_font = font_manager.FontProperties(fname=yahei_font_path)
    for i in range(n_components):
        plt.figure(figsize=(6, 10))
        component_weights = pls_weights.iloc[:, i]
        sorted_indices = component_weights.argsort()
        sorted_weights = component_weights.iloc[sorted_indices]
        sorted_features = sorted_weights.index
        colors = ['blue' if val > 0 else 'red' for val in sorted_weights]
        plt.barh(sorted_features, sorted_weights, color=colors, edgecolor='black')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.title(f'PLS{i + 1} Feature Contributions', fontsize=16, fontweight='bold', fontproperties=custom_font)
        plt.xlabel('Contribution Score', fontsize=14, fontweight='bold', fontproperties=custom_font)
        plt.ylabel('Features', fontsize=14, fontweight='bold', fontproperties=custom_font)
        plt.tight_layout()
        plt.savefig(_output(f'{output_path}\\PLS-DA_{i + 1}_Feature_Contributions.png'))
        plt.savefig(_output(f'{output_path}\\PLS-DA_{i + 1}_Feature_Contributions.svg'))
        plt.show()

def original_072():
    """
Evaluate the original-1DCNN model using ROC curve and prediction-area plot.

All outputs are saved to:
D:\\homework\\夏季学期\\地化大数据\\central asian mapping\\Revised\\fig

Output figures:
1. original-1DCNN_ROC_curve.jpg / .svg
2. original-1DCNN_prediction_area_plot.jpg / .svg

All figure fonts are set to Times New Roman.
"""
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    from sklearn.metrics import roc_curve, auc
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['mathtext.fontset'] = 'stix'
    train_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/grayscale_data_combined_train.csv'
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/grayscale_data_combined_test.csv'
    model_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/1DCNN/trained_model_origin_1DCNN.h5'
    fig_output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Revised/fig'
    if not os.path.exists(fig_output_dir):
        os.makedirs(str(OUT), exist_ok=True)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'没有找到 original-1DCNN 模型文件：{model_path}\n请检查 trained_model_origin_1DCNN.h5 的路径是否正确。')
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    print('测试集CSV文件中的列名如下：')
    print(test_data.columns.tolist())
    feature_cols = train_data.columns[-23:].tolist()
    print('用于 original-1DCNN 模型输入的23个特征列为：')
    print(feature_cols)
    possible_label_cols = ['Label', 'label', 'labels', 'Labels', 'class', 'Class', 'target', 'Target', 'deposit', 'Deposit', 'ore', 'Ore', 'positive', 'Positive', 'mineralization', 'Mineralization', '矿床', '标签', '类别']
    label_col = None
    for col in possible_label_cols:
        if col in test_data.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError('没有找到标签列，无法计算 ROC 曲线和 P-A plot。请检查测试集CSV文件中的真实标签列名。')
    print(f'识别到的标签列为：{label_col}')
    label_numeric = pd.to_numeric(test_data[label_col], errors='coerce')
    valid_label_mask = label_numeric.notna()
    print(f'测试集中总样本数：{len(test_data)}')
    print(f'有效标签样本数：{valid_label_mask.sum()}')
    print(f'无效或空标签样本数：{len(test_data) - valid_label_mask.sum()}')
    if valid_label_mask.sum() == 0:
        raise ValueError('测试集中没有任何有效标签，无法计算 ROC 和 P-A plot。')
    y_true = label_numeric.loc[valid_label_mask].astype(np.int8).to_numpy()
    valid_binary_mask = np.isin(y_true, [0, 1])
    if not np.all(valid_binary_mask):
        print('警告：标签中存在非0/1值，这些样本将被剔除。')
    valid_indices = np.where(valid_label_mask)[0][valid_binary_mask]
    y_true = y_true[valid_binary_mask]
    print(f'用于评价的最终样本数：{len(y_true)}')
    print(f'Number of positive samples: {np.sum(y_true == 1)}')
    print(f'Number of negative samples: {np.sum(y_true == 0)}')
    if len(np.unique(y_true)) < 2:
        raise ValueError('用于评价的测试样本中只有一种标签，无法计算 ROC 曲线和 AUC。请检查测试集是否同时包含正样本和负样本。')
    missing_features = [col for col in feature_cols if col not in test_data.columns]
    if len(missing_features) > 0:
        raise ValueError(f'测试集中缺少以下特征列：{missing_features}')
    # Map the full study area. Keep the ROC and P-A evaluation on labelled pixels.
    features = test_data[feature_cols].to_numpy(dtype=np.float32, copy=True)
    model = load_model(model_path)
    predictions = model.predict(features.reshape(-1, 23, 1), batch_size=8192, verbose=1)
    if predictions.ndim == 2 and predictions.shape[1] == 2:
        full_scores = predictions[:, 1]
    else:
        full_scores = predictions.ravel()
    full_scores = full_scores.astype(np.float32, copy=False)
    export_prediction_map(test_data, full_scores, 'Original_1DCNN')
    y_score = full_scores[valid_indices]
    print(f'Prediction score shape: {y_score.shape}')
    prediction_df = pd.DataFrame()
    if 'x' in test_data.columns:
        prediction_df['x'] = test_data.loc[test_data.index[valid_indices], 'x'].to_numpy()
    if 'y' in test_data.columns:
        prediction_df['y'] = test_data.loc[test_data.index[valid_indices], 'y'].to_numpy()
    prediction_df[label_col] = y_true
    prediction_df['original-1DCNN_Predicted_Probability'] = y_score
    prediction_csv_path = os.path.join(fig_output_dir, 'original-1DCNN_predictions.csv')
    prediction_df.to_csv(_output(prediction_csv_path), index=False)
    print(f'预测结果已保存为：{prediction_csv_path}')
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    export_roc_auc('Original_1DCNN', fpr, tpr, thresholds, roc_auc, y_true, 'labelled pixels')
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'original-1DCNN, AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, label='Random prediction')
    plt.xlabel('False positive rate', fontsize=14)
    plt.ylabel('True positive rate', fontsize=14)
    plt.title('ROC Curve of the original-1DCNN Model', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    roc_jpg_path = os.path.join(fig_output_dir, 'original-1DCNN_ROC_curve.jpg')
    roc_svg_path = os.path.join(fig_output_dir, 'original-1DCNN_ROC_curve.svg')
    plt.savefig(_output(roc_jpg_path), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(_output(roc_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'ROC曲线已保存为：{roc_jpg_path}')
    print(f'ROC曲线已保存为：{roc_svg_path}')
    total_positive = np.sum(y_true == 1)
    if total_positive == 0:
        raise ValueError('测试集中没有正样本，无法绘制 P-A plot。')
    score_min = np.min(y_score)
    score_max = np.max(y_score)
    if score_min >= 0 and score_max <= 1:
        bins = np.linspace(0, 1, 11)
    else:
        bins = np.linspace(score_min, score_max, 11)
    class_labels = []
    area_percent_by_class = []
    captured_percent_by_class = []
    prediction_efficiency = []
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        if i == len(bins) - 2:
            class_mask = (y_score >= lower) & (y_score <= upper)
        else:
            class_mask = (y_score >= lower) & (y_score < upper)
        area_percent = np.sum(class_mask) / len(y_score) * 100
        captured_percent = np.sum((y_true == 1) & class_mask) / total_positive * 100
        if area_percent > 0:
            efficiency = captured_percent / area_percent
        else:
            efficiency = np.nan
        class_labels.append(f'{lower:.2f}-{upper:.2f}')
        area_percent_by_class.append(area_percent)
        captured_percent_by_class.append(captured_percent)
        prediction_efficiency.append(efficiency)
    class_labels = class_labels[::-1]
    area_percent_by_class = area_percent_by_class[::-1]
    captured_percent_by_class = captured_percent_by_class[::-1]
    prediction_efficiency = prediction_efficiency[::-1]
    x = np.arange(len(class_labels))
    width = 0.38
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, area_percent_by_class, width, label='original-1DCNN predicted area (%)', edgecolor='black')
    plt.bar(x + width / 2, captured_percent_by_class, width, label='original-1DCNN captured positive samples (%)', edgecolor='black')
    plt.xlabel('Prospectivity score class', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.title('Prediction-Area Plot of the original-1DCNN Model', fontsize=16)
    plt.xticks(x, class_labels, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y')
    plt.tight_layout()
    pa_jpg_path = os.path.join(fig_output_dir, 'original-1DCNN_prediction_area_plot.jpg')
    pa_svg_path = os.path.join(fig_output_dir, 'original-1DCNN_prediction_area_plot.svg')
    plt.savefig(_output(pa_jpg_path), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(_output(pa_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'P-A plot 已保存为：{pa_jpg_path}')
    print(f'P-A plot 已保存为：{pa_svg_path}')
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_labels = y_true[sorted_indices]
    sorted_scores = y_score[sorted_indices]
    cumulative_area = np.arange(1, len(sorted_labels) + 1) / len(sorted_labels) * 100
    cumulative_deposits = np.cumsum(sorted_labels == 1) / total_positive * 100
    capture_results = []
    for area_threshold in [5, 10, 20, 30, 40, 50]:
        idx = np.searchsorted(cumulative_area, area_threshold)
        if idx < len(cumulative_deposits):
            captured = cumulative_deposits[idx]
        else:
            captured = 100
        capture_results.append([area_threshold, captured])
        print(f'Top {area_threshold}% predicted area captures {captured:.2f}% of positive samples.')
    capture_df = pd.DataFrame(capture_results, columns=['Predicted_area_percent', 'Captured_positive_samples_percent'])
    capture_csv_path = os.path.join(fig_output_dir, 'original-1DCNN_capture_results.csv')
    capture_df.to_csv(_output(capture_csv_path), index=False)
    roc_data_df = pd.DataFrame({'False_positive_rate': fpr, 'True_positive_rate': tpr, 'Threshold': thresholds})
    roc_csv_path = os.path.join(fig_output_dir, 'original-1DCNN_ROC_curve_data.csv')
    roc_data_df.to_csv(_output(roc_csv_path), index=False)
    pa_data_df = pd.DataFrame({'Prospectivity_score_class': class_labels, 'Predicted_area_percent': area_percent_by_class, 'Captured_positive_samples_percent': captured_percent_by_class, 'Prediction_efficiency': prediction_efficiency})
    pa_csv_path = os.path.join(fig_output_dir, 'original-1DCNN_prediction_area_plot_data.csv')
    pa_data_df.to_csv(_output(pa_csv_path), index=False)
    metrics_txt_path = os.path.join(fig_output_dir, 'original-1DCNN_model_evaluation_metrics.txt')
    with open(_output(metrics_txt_path), 'w', encoding='utf-8') as f:
        f.write('original-1DCNN model evaluation results\n')
        f.write('======================================\n')
        f.write(f'AUC = {roc_auc:.4f}\n')
        f.write(f'Total evaluation samples = {len(y_true)}\n')
        f.write(f'Positive samples = {np.sum(y_true == 1)}\n')
        f.write(f'Negative samples = {np.sum(y_true == 0)}\n\n')
        f.write('Prediction-area plot results:\n')
        for label, area, captured, efficiency in zip(class_labels, area_percent_by_class, captured_percent_by_class, prediction_efficiency):
            f.write(f'{label}: predicted area = {area:.2f}%, captured positive samples = {captured:.2f}%, prediction efficiency = {efficiency:.2f}\n')
        f.write('\nPrediction-area capture results:\n')
        for area_threshold, captured in capture_results:
            f.write(f'Top {area_threshold}% predicted area captures {captured:.2f}% of positive samples.\n')
    print(f'预测结果已保存为：{prediction_csv_path}')
    print(f'捕获效率结果已保存为：{capture_csv_path}')
    print(f'ROC数据已保存为：{roc_csv_path}')
    print(f'P-A plot 数据已保存为：{pa_csv_path}')
    print(f'模型评价指标已保存为：{metrics_txt_path}')
    print('全部分析完成。')
    export_pa_summary(pa_data_df, 'Original_1DCNN', 'labelled pixels only')

def original_073():
    """
Evaluate the original 2DCNN model using ROC curve and prediction-area plot.

All outputs are saved to:
D:\\homework\\夏季学期\\地化大数据\\central asian mapping\\Revised\x0cig

Output figures:
1. original-2DCNN_ROC_curve.jpg / .svg
2. original-2DCNN_prediction_area_plot.jpg / .svg

All figure fonts are set to Times New Roman.
"""
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    from sklearn.metrics import roc_curve, auc
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['mathtext.fontset'] = 'stix'
    model_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/2DCNN_model_original_analyse.h5'
    train_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/grayscale_data_combined_train.csv'
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/grayscale_data_combined_test.csv'
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Revised/fig'
    if not os.path.exists(output_dir):
        os.makedirs(str(OUT), exist_ok=True)
    pca_data_train = pd.read_csv(train_file_path)
    pca_data_test = pd.read_csv(test_file_path)
    print('测试集CSV文件中的列名如下：')
    print(pca_data_test.columns.tolist())
    possible_x_cols = ['x', 'X', 'lon', 'Lon', 'longitude', 'Longitude']
    possible_y_cols = ['y', 'Y', 'lat', 'Lat', 'latitude', 'Latitude']
    x_col = None
    y_col = None
    for col in possible_x_cols:
        if col in pca_data_test.columns:
            x_col = col
            break
    for col in possible_y_cols:
        if col in pca_data_test.columns:
            y_col = col
            break
    if x_col is None or y_col is None:
        raise ValueError('没有找到 x 或 y 坐标列。请检查CSV中的坐标列名，并在 possible_x_cols 或 possible_y_cols 中补充。')
    print(f'识别到的X坐标列为：{x_col}')
    print(f'识别到的Y坐标列为：{y_col}')
    x_coords = pca_data_test[x_col].values
    y_coords = pca_data_test[y_col].values
    possible_label_cols = ['label', 'Label', 'labels', 'Labels', 'class', 'Class', 'target', 'Target', 'deposit', 'Deposit', 'ore', 'Ore', 'positive', 'Positive', 'mineralization', 'Mineralization', '矿床', '标签', '类别']
    label_col = None
    for col in possible_label_cols:
        if col in pca_data_test.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError('没有找到标签列，无法计算ROC曲线和P-A plot。请检查CSV文件中的真实标签列名，并将其加入 possible_label_cols。')
    print(f'识别到的标签列为：{label_col}')
    label_values = pca_data_test[label_col].values
    label_values = np.asarray(label_values).astype(int)
    exclude_cols = [x_col, y_col, label_col]
    feature_df = pca_data_test.drop(columns=exclude_cols)
    feature_df = feature_df.select_dtypes(include=[np.number])
    if feature_df.shape[1] < 23:
        raise ValueError(f'可用数值特征列少于23列，目前只有 {feature_df.shape[1]} 列。')
    feature_cols = feature_df.columns[-23:].tolist()
    print('用于 original 2DCNN 模型输入的23个特征列为：')
    print(feature_cols)
    feature_values = pca_data_test[feature_cols].values
    print(f'Feature shape: {feature_values.shape}')
    print(f'Label shape: {label_values.shape}')

    def prepare_test_data_by_coordinates_with_labels(x_coords, y_coords, feature_values, label_values, patch_size=9, min_points=50):
        """
    生成 9 × 9 × 23 的测试patch，并为每个patch生成对应标签。

    如果patch中包含至少一个正样本像元，则该patch标签为1；
    如果patch中不包含正样本像元，则该patch标签为0。
    """
        reshaped_patches = []
        patch_labels = []
        indices = []
        min_x, max_x = (min(x_coords), max(x_coords))
        min_y, max_y = (min(y_coords), max(y_coords))
        x_steps = np.arange(min_x, max_x + 1, patch_size)
        y_steps = np.arange(min_y, max_y + 1, patch_size)
        for y_min in y_steps:
            y_max = y_min + patch_size - 1
            if y_max > max_y:
                continue
            for x_min in x_steps:
                x_max = x_min + patch_size - 1
                if x_max > max_x:
                    continue
                mask = (x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)
                patch_data = feature_values[mask]
                patch_label_data = label_values[mask]
                patch_x = x_coords[mask]
                patch_y = y_coords[mask]
                num_points = len(patch_data)
                if num_points >= min_points:
                    sort_order = np.lexsort((patch_x, patch_y))
                    patch_data = patch_data[sort_order]
                    patch_label_data = patch_label_data[sort_order]
                    patch_label = 1 if np.any(patch_label_data == 1) else 0
                    if num_points < patch_size * patch_size:
                        patch_data = np.pad(patch_data, ((0, patch_size * patch_size - num_points), (0, 0)), mode='constant')
                    try:
                        reshaped_patch = patch_data.reshape(patch_size, patch_size, -1)
                        reshaped_patches.append(reshaped_patch)
                        patch_labels.append(patch_label)
                        indices.append((x_min, x_max, y_min, y_max))
                    except ValueError as e:
                        print(f'Error reshaping patch at X: {x_min}-{x_max}, Y: {y_min}-{y_max} -> {e}')
                else:
                    print(f'Skipping patch at X: {x_min}-{x_max}, Y: {y_min}-{y_max}, insufficient points.')
        print(f'Total patches generated: {len(reshaped_patches)}')
        return (np.array(reshaped_patches), np.array(patch_labels), indices)
    test_patches, y_true, indices = prepare_test_data_by_coordinates_with_labels(x_coords, y_coords, feature_values, label_values, patch_size=9, min_points=50)
    print(f'Test patches shape: {test_patches.shape}')
    print(f'Patch labels shape: {y_true.shape}')
    print(f'Number of positive patches: {np.sum(y_true == 1)}')
    print(f'Number of negative patches: {np.sum(y_true == 0)}')
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        raise ValueError('测试patch中只有一种标签，无法计算ROC曲线和AUC。请检查测试集是否同时包含正样本和负样本。')
    model = load_model(model_path)
    print(f'模型已加载：{model_path}')
    print(f'输出目录：{output_dir}')
    predictions = model.predict(test_patches)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        y_score = predictions.ravel()
    elif predictions.ndim == 2 and predictions.shape[1] == 2:
        y_score = predictions[:, 1]
    else:
        y_score = predictions.ravel()
    print(f'Prediction score shape: {y_score.shape}')
    pca_data_test['original-2DCNN_Predicted_Probability'] = np.nan
    for idx, (x_min, x_max, y_min, y_max) in enumerate(indices):
        mask = (x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)
        pca_data_test.loc[mask, 'original-2DCNN_Predicted_Probability'] = y_score[idx]
    prediction_csv_path = os.path.join(output_dir, 'original-2DCNN_predictions.csv')
    pca_data_test.to_csv(_output(prediction_csv_path), index=False)
    print(f'预测结果已保存为：{prediction_csv_path}')
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    export_roc_auc('Original_2DCNN', fpr, tpr, thresholds, roc_auc, y_true, '9x9 patches')
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'original-2DCNN, AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, label='Random prediction')
    plt.xlabel('False positive rate', fontsize=14)
    plt.ylabel('True positive rate', fontsize=14)
    plt.title('ROC Curve of the original-2DCNN Model', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    roc_jpg_path = os.path.join(output_dir, 'original-2DCNN_ROC_curve.jpg')
    roc_svg_path = os.path.join(output_dir, 'original-2DCNN_ROC_curve.svg')
    plt.savefig(_output(roc_jpg_path), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(_output(roc_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'ROC曲线已保存为：{roc_jpg_path}')
    print(f'ROC曲线已保存为：{roc_svg_path}')
    total_positive = np.sum(y_true == 1)
    if total_positive == 0:
        raise ValueError('测试集中没有正样本，无法绘制 P-A plot。')
    score_min = np.min(y_score)
    score_max = np.max(y_score)
    if score_min >= 0 and score_max <= 1:
        bins = np.linspace(0, 1, 11)
    else:
        bins = np.linspace(score_min, score_max, 11)
    class_labels = []
    area_percent_by_class = []
    captured_percent_by_class = []
    prediction_efficiency = []
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        if i == len(bins) - 2:
            class_mask = (y_score >= lower) & (y_score <= upper)
        else:
            class_mask = (y_score >= lower) & (y_score < upper)
        area_percent = np.sum(class_mask) / len(y_score) * 100
        captured_percent = np.sum((y_true == 1) & class_mask) / total_positive * 100
        if area_percent > 0:
            efficiency = captured_percent / area_percent
        else:
            efficiency = np.nan
        class_labels.append(f'{lower:.2f}-{upper:.2f}')
        area_percent_by_class.append(area_percent)
        captured_percent_by_class.append(captured_percent)
        prediction_efficiency.append(efficiency)
    class_labels = class_labels[::-1]
    area_percent_by_class = area_percent_by_class[::-1]
    captured_percent_by_class = captured_percent_by_class[::-1]
    prediction_efficiency = prediction_efficiency[::-1]
    x = np.arange(len(class_labels))
    width = 0.38
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, area_percent_by_class, width, label='original-2DCNN predicted area (%)', edgecolor='black')
    plt.bar(x + width / 2, captured_percent_by_class, width, label='original-2DCNN captured positive samples (%)', edgecolor='black')
    plt.xlabel('Prospectivity score class', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.title('Prediction-Area Plot of the original-2DCNN Model', fontsize=16)
    plt.xticks(x, class_labels, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y')
    plt.tight_layout()
    pa_jpg_path = os.path.join(output_dir, 'original-2DCNN_prediction_area_plot.jpg')
    pa_svg_path = os.path.join(output_dir, 'original-2DCNN_prediction_area_plot.svg')
    plt.savefig(_output(pa_jpg_path), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(_output(pa_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'P-A plot 已保存为：{pa_jpg_path}')
    print(f'P-A plot 已保存为：{pa_svg_path}')
    roc_data_df = pd.DataFrame({'False_positive_rate': fpr, 'True_positive_rate': tpr, 'Threshold': thresholds})
    roc_csv_path = os.path.join(output_dir, 'original-2DCNN_ROC_curve_data.csv')
    roc_data_df.to_csv(_output(roc_csv_path), index=False)
    pa_data_df = pd.DataFrame({'Prospectivity_score_class': class_labels, 'Predicted_area_percent': area_percent_by_class, 'Captured_positive_samples_percent': captured_percent_by_class, 'Prediction_efficiency': prediction_efficiency})
    pa_csv_path = os.path.join(output_dir, 'original-2DCNN_prediction_area_plot_data.csv')
    pa_data_df.to_csv(_output(pa_csv_path), index=False)
    metrics_txt_path = os.path.join(output_dir, 'original-2DCNN_model_evaluation_metrics.txt')
    with open(_output(metrics_txt_path), 'w', encoding='utf-8') as f:
        f.write('original-2DCNN model evaluation results\n')
        f.write('======================================\n')
        f.write(f'AUC = {roc_auc:.4f}\n')
        f.write(f'Total patches = {len(y_true)}\n')
        f.write(f'Positive patches = {np.sum(y_true == 1)}\n')
        f.write(f'Negative patches = {np.sum(y_true == 0)}\n\n')
        f.write('Prediction-area plot results:\n')
        for label, area, captured, efficiency in zip(class_labels, area_percent_by_class, captured_percent_by_class, prediction_efficiency):
            f.write(f'{label}: predicted area = {area:.2f}%, captured positive samples = {captured:.2f}%, prediction efficiency = {efficiency:.2f}\n')
    print(f'ROC数据已保存为：{roc_csv_path}')
    print(f'P-A plot 数据已保存为：{pa_csv_path}')
    print(f'模型评价指标已保存为：{metrics_txt_path}')
    print('全部分析完成。')
    export_pa_summary(pa_data_df, 'Original_2DCNN', 'nonoverlapping 9x9 patches; any positive pixel makes positive patch')

def original_074():
    """
Evaluate the original-RF model using ROC curve and prediction-area plot.

This script follows the logic of original_RF_test.py:
1. Load training and test data
2. Train Random Forest using RandomizedSearchCV
3. Use the trained best_model to predict probabilities
4. Plot ROC curve and Prediction-Area plot

All outputs are saved to:
D:\\homework\\夏季学期\\地化大数据\\central asian mapping\\Revised\\fig

Output figures:
1. original-RF_ROC_curve.jpg / .svg
2. original-RF_prediction_area_plot.jpg / .svg

All figure fonts are set to Times New Roman.
"""
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import roc_curve, auc
    import joblib
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['mathtext.fontset'] = 'stix'
    train_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/grayscale_data_combined_train.csv'
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/original_analyses/grayscale_data_combined_test.csv'
    fig_output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Revised/fig'
    if not os.path.exists(fig_output_dir):
        os.makedirs(str(OUT), exist_ok=True)
    pca_data_train = pd.read_csv(train_file_path)
    pca_data_test = pd.read_csv(test_file_path)
    print('训练集CSV文件中的列名如下：')
    print(pca_data_train.columns.tolist())
    print('测试集CSV文件中的列名如下：')
    print(pca_data_test.columns.tolist())
    feature_cols = pca_data_train.loc[:, '176Hf-177Hf':'Hf(t)'].columns.tolist()
    print('用于 original-RF 模型输入的特征列为：')
    print(feature_cols)
    print(f'特征数量：{len(feature_cols)}')
    X_train = pca_data_train[feature_cols].values.astype(np.float32)
    y_train = pca_data_train['Label'].values
    X_test = pca_data_test[feature_cols].values.astype(np.float32)
    train_label_numeric = pd.to_numeric(y_train, errors='coerce')
    valid_train_mask = ~pd.isna(train_label_numeric) & np.isin(train_label_numeric, [0, 1])
    X_train = X_train[valid_train_mask]
    y_train = train_label_numeric[valid_train_mask].astype(int)
    print(f'用于训练的样本数：{len(y_train)}')
    print(f'训练集正样本数：{np.sum(y_train == 1)}')
    print(f'训练集负样本数：{np.sum(y_train == 0)}')
    if len(np.unique(y_train)) < 2:
        raise ValueError('训练集中只有一种标签，无法训练二分类Random Forest模型。')
    if not np.isfinite(X_train).all():
        print('训练特征中存在NaN或无穷值，已替换为0。')
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not np.isfinite(X_test).all():
        print('测试特征中存在NaN或无穷值，已替换为0。')
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    param_dist = {'n_estimators': np.arange(2, 500), 'max_depth': np.arange(2, 31), 'min_samples_split': [2], 'min_samples_leaf': [1, 2, 4], 'bootstrap': [True, False]}
    random_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=0), param_distributions=param_dist, n_iter=10, cv=3, verbose=2, n_jobs=-1, random_state=0)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    print('最佳参数组合：')
    print(best_params)
    best_params_txt_path = os.path.join(fig_output_dir, 'original-RF_best_params.txt')
    with open(_output(best_params_txt_path), 'w', encoding='utf-8') as f:
        for param, value in best_params.items():
            f.write(f'{param}: {value}\n')
    print(f'最佳参数已保存为：{best_params_txt_path}')
    best_model = random_search.best_estimator_
    rf_model_path = os.path.join(fig_output_dir, 'original-RF_trained_model.joblib')
    joblib.dump(best_model, _output(rf_model_path))
    print(f'训练好的original-RF模型已保存为：{rf_model_path}')
    y_score_all = best_model.predict_proba(X_test)[:, 1]
    y_score_all = y_score_all.astype(np.float32)
    print(f'预测概率数量：{len(y_score_all)}')
    print(f'最低预测概率：{np.min(y_score_all):.6f}')
    print(f'最高预测概率：{np.max(y_score_all):.6f}')
    export_prediction_map(pca_data_test, y_score_all, 'Original_RF')
    pca_data_test['original-RF_Predicted_Probability'] = y_score_all
    prediction_csv_path = os.path.join(fig_output_dir, 'original-RF_predictions.csv')
    pca_data_test.to_csv(_output(prediction_csv_path), index=False)
    print(f'预测结果已保存为：{prediction_csv_path}')
    test_label_numeric = pd.to_numeric(pca_data_test['Label'], errors='coerce')
    valid_label_mask = test_label_numeric.notna() & test_label_numeric.isin([0, 1])
    valid_label_mask_array = valid_label_mask.to_numpy()
    print(f'测试集总样本数：{len(pca_data_test)}')
    print(f'有效0/1标签样本数：{np.sum(valid_label_mask_array)}')
    print(f'空标签或非0/1标签样本数：{len(pca_data_test) - np.sum(valid_label_mask_array)}')
    if np.sum(valid_label_mask_array) == 0:
        raise ValueError('测试集中没有有效的0/1标签，无法计算ROC曲线和P-A plot。')
    y_true = test_label_numeric.loc[valid_label_mask].astype(int).to_numpy()
    y_score_roc = y_score_all[valid_label_mask_array]
    print(f'用于ROC评价的样本数：{len(y_true)}')
    print(f'测试集正样本数：{np.sum(y_true == 1)}')
    print(f'测试集负样本数：{np.sum(y_true == 0)}')
    if len(np.unique(y_true)) < 2:
        raise ValueError('有效标签样本中只有一种类别，无法计算ROC曲线和AUC。')
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score_roc)
    roc_auc = auc(fpr, tpr)
    export_roc_auc('Original_RF', fpr, tpr, roc_thresholds, roc_auc, y_true, 'labelled pixels')
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'original-RF, AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, label='Random prediction')
    plt.xlabel('False positive rate', fontsize=14)
    plt.ylabel('True positive rate', fontsize=14)
    plt.title('ROC Curve of the original-RF Model', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    roc_jpg_path = os.path.join(fig_output_dir, 'original-RF_ROC_curve.jpg')
    roc_svg_path = os.path.join(fig_output_dir, 'original-RF_ROC_curve.svg')
    plt.savefig(_output(roc_jpg_path), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(_output(roc_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'ROC曲线已保存为：{roc_jpg_path}')
    print(f'ROC曲线已保存为：{roc_svg_path}')
    positive_global_mask = np.zeros(len(y_score_all), dtype=bool)
    positive_global_mask[valid_label_mask_array] = y_true == 1
    total_positive = int(np.sum(positive_global_mask))
    if total_positive == 0:
        raise ValueError('测试集中没有有效正样本，无法绘制P-A plot。')
    score_min = float(np.min(y_score_all))
    score_max = float(np.max(y_score_all))
    if score_min >= 0 and score_max <= 1:
        bins = np.linspace(0, 1, 11)
    else:
        bins = np.linspace(score_min, score_max, 11)
    class_labels = []
    area_percent_by_class = []
    captured_percent_by_class = []
    prediction_efficiency = []
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        if i == len(bins) - 2:
            class_mask = (y_score_all >= lower) & (y_score_all <= upper)
        else:
            class_mask = (y_score_all >= lower) & (y_score_all < upper)
        area_percent = np.sum(class_mask) / len(y_score_all) * 100
        captured_percent = np.sum(positive_global_mask & class_mask) / total_positive * 100
        if area_percent > 0:
            efficiency = captured_percent / area_percent
        else:
            efficiency = np.nan
        class_labels.append(f'{lower:.2f}-{upper:.2f}')
        area_percent_by_class.append(float(area_percent))
        captured_percent_by_class.append(float(captured_percent))
        prediction_efficiency.append(float(efficiency) if np.isfinite(efficiency) else np.nan)
    class_labels = class_labels[::-1]
    area_percent_by_class = area_percent_by_class[::-1]
    captured_percent_by_class = captured_percent_by_class[::-1]
    prediction_efficiency = prediction_efficiency[::-1]
    x_positions = np.arange(len(class_labels))
    bar_width = 0.38
    plt.figure(figsize=(10, 6))
    plt.bar(x_positions - bar_width / 2, area_percent_by_class, bar_width, label='original-RF predicted area (%)', edgecolor='black')
    plt.bar(x_positions + bar_width / 2, captured_percent_by_class, bar_width, label='original-RF captured positive samples (%)', edgecolor='black')
    plt.xlabel('Prospectivity score class', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.title('Prediction-Area Plot of the original-RF Model', fontsize=16)
    plt.xticks(x_positions, class_labels, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y')
    plt.tight_layout()
    pa_jpg_path = os.path.join(fig_output_dir, 'original-RF_prediction_area_plot.jpg')
    pa_svg_path = os.path.join(fig_output_dir, 'original-RF_prediction_area_plot.svg')
    plt.savefig(_output(pa_jpg_path), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(_output(pa_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'P-A plot已保存为：{pa_jpg_path}')
    print(f'P-A plot已保存为：{pa_svg_path}')
    sorted_indices = np.argsort(y_score_all)[::-1]
    sorted_positive_labels = positive_global_mask[sorted_indices]
    sorted_scores = y_score_all[sorted_indices]
    cumulative_area = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    cumulative_positive_samples = np.cumsum(sorted_positive_labels) / total_positive * 100
    capture_results = []
    for area_threshold in [5, 10, 20, 30, 40, 50]:
        idx = np.searchsorted(cumulative_area, area_threshold)
        if idx < len(cumulative_positive_samples):
            captured = cumulative_positive_samples[idx]
        else:
            captured = 100.0
        capture_results.append([area_threshold, captured])
        print(f'Top {area_threshold}% predicted area captures {captured:.2f}% of positive samples.')
    roc_data_df = pd.DataFrame({'False_positive_rate': fpr, 'True_positive_rate': tpr, 'Threshold': roc_thresholds})
    roc_csv_path = os.path.join(fig_output_dir, 'original-RF_ROC_curve_data.csv')
    roc_data_df.to_csv(_output(roc_csv_path), index=False)
    pa_data_df = pd.DataFrame({'Prospectivity_score_class': class_labels, 'Predicted_area_percent': area_percent_by_class, 'Captured_positive_samples_percent': captured_percent_by_class, 'Prediction_efficiency': prediction_efficiency})
    pa_csv_path = os.path.join(fig_output_dir, 'original-RF_prediction_area_plot_data.csv')
    pa_data_df.to_csv(_output(pa_csv_path), index=False)
    capture_df = pd.DataFrame(capture_results, columns=['Predicted_area_percent', 'Captured_positive_samples_percent'])
    capture_csv_path = os.path.join(fig_output_dir, 'original-RF_capture_results.csv')
    capture_df.to_csv(_output(capture_csv_path), index=False)
    metrics_txt_path = os.path.join(fig_output_dir, 'original-RF_model_evaluation_metrics.txt')
    with open(_output(metrics_txt_path), 'w', encoding='utf-8') as f:
        f.write('original-RF model evaluation results\n')
        f.write('====================================\n')
        f.write(f'AUC = {roc_auc:.4f}\n')
        f.write(f'Total predicted samples = {len(y_score_all)}\n')
        f.write(f'Total labelled samples = {len(y_true)}\n')
        f.write(f'Positive samples = {np.sum(y_true == 1)}\n')
        f.write(f'Negative samples = {np.sum(y_true == 0)}\n\n')
        f.write('Prediction-area plot results:\n')
        for score_class, area, captured, efficiency in zip(class_labels, area_percent_by_class, captured_percent_by_class, prediction_efficiency):
            f.write(f'{score_class}: predicted area = {area:.2f}%, captured positive samples = {captured:.2f}%, prediction efficiency = {efficiency:.2f}\n')
        f.write('\nPrediction-area capture results:\n')
        for area_threshold, captured in capture_results:
            f.write(f'Top {area_threshold}% predicted area captures {captured:.2f}% of positive samples.\n')
    feature_contributions = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Gini_importance': feature_contributions})
    feature_importance_csv_path = os.path.join(fig_output_dir, 'original-RF_feature_importance.csv')
    feature_importance_df.to_csv(_output(feature_importance_csv_path), index=False)
    print(f'ROC数据已保存为：{roc_csv_path}')
    print(f'P-A plot数据已保存为：{pa_csv_path}')
    print(f'捕获效率结果已保存为：{capture_csv_path}')
    print(f'模型评价指标已保存为：{metrics_txt_path}')
    print(f'特征重要性结果已保存为：{feature_importance_csv_path}')
    print('全部分析完成。')
    export_pa_summary(pa_data_df, 'Original_RF', 'labelled pixels only')

def original_075():
    """
Created on Mon Jul 13 17:16:00 2026

@author: ASUS
"""
    '\nEvaluate the PCA-1DCNN model using ROC curve and prediction-area plot.\n\nAll outputs are saved to:\nD:\\homework\\夏季学期\\地化大数据\\central asian mapping\\Revised\\fig\n\nOutput figures:\n1. PCA-1DCNN_ROC_curve.jpg / .svg\n2. PCA-1DCNN_prediction_area_plot.jpg / .svg\n\nAll figure fonts are set to Times New Roman.\n'
    import os
    import gc
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    from sklearn.metrics import roc_curve, auc
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['mathtext.fontset'] = 'stix'
    model_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/1DCNN_test/ensure/trained_model_1DCNN_test.h5'
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/PCA_with_labels_Test.csv'
    fig_output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Revised/fig'
    os.makedirs(str(OUT), exist_ok=True)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'没有找到PCA-1DCNN模型文件：\n{model_path}\n请检查模型路径是否正确。')
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f'没有找到测试数据文件：\n{test_file_path}\n请检查测试数据路径是否正确。')
    pca_columns = [f'PCA{i}' for i in range(1, 24)]
    possible_label_cols = ['Label', 'label', 'Labels', 'labels', 'Class', 'class', 'Target', 'target', 'Deposit', 'deposit', 'Ore', 'ore', 'Positive', 'positive', 'Mineralization', 'mineralization', '标签', '类别']
    possible_x_cols = ['x', 'X', 'lon', 'Lon', 'longitude', 'Longitude']
    possible_y_cols = ['y', 'Y', 'lat', 'Lat', 'latitude', 'Latitude']
    header_columns = pd.read_csv(test_file_path, nrows=0).columns.tolist()
    print('测试集CSV文件中的列名如下：')
    print(header_columns)
    missing_pca_columns = [col for col in pca_columns if col not in header_columns]
    if missing_pca_columns:
        raise ValueError(f'测试集中缺少以下PCA特征列：\n{missing_pca_columns}')
    label_col = None
    for col in possible_label_cols:
        if col in header_columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError('没有找到标签列，无法计算ROC曲线和P-A plot。\n请检查CSV文件中的真实标签列名。')
    print(f'识别到的标签列为：{label_col}')
    x_col = None
    y_col = None
    for col in possible_x_cols:
        if col in header_columns:
            x_col = col
            break
    for col in possible_y_cols:
        if col in header_columns:
            y_col = col
            break
    if x_col is not None:
        print(f'识别到的X坐标列为：{x_col}')
    else:
        print('未识别到X坐标列。')
    if y_col is not None:
        print(f'识别到的Y坐标列为：{y_col}')
    else:
        print('未识别到Y坐标列。')
    use_columns = pca_columns + [label_col]
    if x_col is not None:
        use_columns.append(x_col)
    if y_col is not None:
        use_columns.append(y_col)
    use_columns = list(dict.fromkeys(use_columns))
    dtype_dict = {col: np.float32 for col in pca_columns}
    if x_col is not None:
        dtype_dict[x_col] = np.float32
    if y_col is not None:
        dtype_dict[y_col] = np.float32
    dtype_dict[label_col] = np.float32
    test_data = pd.read_csv(test_file_path, usecols=use_columns, dtype=dtype_dict)
    print(f'测试集总行数：{len(test_data)}')
    test_pca_values = test_data[pca_columns].to_numpy(dtype=np.float32, copy=False)
    print(f'PCA feature shape before reshape: {test_pca_values.shape}')
    invalid_feature_rows = ~np.isfinite(test_pca_values).all(axis=1)
    invalid_feature_count = int(np.sum(invalid_feature_rows))
    print(f'包含NaN或无穷值的特征行数：{invalid_feature_count}')
    if invalid_feature_count > 0:
        print('警告：PCA特征中存在NaN或无穷值，将这些值替换为0。')
        test_pca_values = np.nan_to_num(test_pca_values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    test_image = test_pca_values.reshape(-1, 23, 1)
    print(f'PCA-1DCNN input shape: {test_image.shape}')
    model = load_model(model_path)
    print(f'模型已加载：{model_path}')
    print(f'模型输入形状：{model.input_shape}')
    print(f'图片输出目录：{fig_output_dir}')
    expected_input_shape = model.input_shape
    if isinstance(expected_input_shape, list):
        expected_input_shape = expected_input_shape[0]
    if len(expected_input_shape) == 3 and expected_input_shape[1] is not None and (expected_input_shape[1] != 23):
        raise ValueError(f'模型要求的输入长度为 {expected_input_shape[1]}，但当前输入特征长度为23。')
    predictions = model.predict(test_image, batch_size=8192, verbose=1)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        y_score_all = predictions[:, 0]
    elif predictions.ndim == 2 and predictions.shape[1] == 2:
        y_score_all = predictions[:, 1]
    else:
        y_score_all = predictions.reshape(-1)
    y_score_all = y_score_all.astype(np.float32, copy=False)
    print(f'所有预测概率的形状：{y_score_all.shape}')
    print(f'最低预测概率：{np.min(y_score_all):.6f}')
    print(f'最高预测概率：{np.max(y_score_all):.6f}')
    label_numeric = pd.to_numeric(test_data[label_col], errors='coerce')
    valid_label_mask = label_numeric.notna() & label_numeric.isin([0, 1])
    valid_label_mask_array = valid_label_mask.to_numpy()
    print(f'有效0/1标签样本数：{int(np.sum(valid_label_mask_array))}')
    print(f'空标签或非0/1标签样本数：{len(test_data) - int(np.sum(valid_label_mask_array))}')
    if np.sum(valid_label_mask_array) == 0:
        raise ValueError('测试集中没有有效的0/1标签，无法计算ROC曲线和P-A plot。')
    y_true = label_numeric.loc[valid_label_mask].astype(np.int8).to_numpy()
    y_score_roc = y_score_all[valid_label_mask_array]
    print(f'用于ROC评价的样本数：{len(y_true)}')
    print(f'正样本数：{np.sum(y_true == 1)}')
    print(f'负样本数：{np.sum(y_true == 0)}')
    if len(np.unique(y_true)) < 2:
        raise ValueError('有效标签样本中只有一个类别，无法计算ROC曲线和AUC。\n请检查测试集是否同时包含正样本和负样本。')
    prediction_output = {}
    if x_col is not None:
        prediction_output[x_col] = test_data[x_col].to_numpy()
    if y_col is not None:
        prediction_output[y_col] = test_data[y_col].to_numpy()
    prediction_output[label_col] = test_data[label_col].to_numpy()
    prediction_output['PCA-1DCNN_Predicted_Probability'] = y_score_all
    prediction_df = pd.DataFrame(prediction_output)
    prediction_csv_path = os.path.join(fig_output_dir, 'PCA-1DCNN_predictions.csv')
    prediction_df.to_csv(_output(prediction_csv_path), index=False)
    print(f'预测结果已保存为：{prediction_csv_path}')
    del predictions
    del test_image
    del test_pca_values
    gc.collect()
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score_roc)
    roc_auc = auc(fpr, tpr)
    export_roc_auc('PCA_1DCNN', fpr, tpr, roc_thresholds, roc_auc, y_true, 'labelled pixels')
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'PCA-1DCNN, AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, label='Random prediction')
    plt.xlabel('False positive rate', fontsize=14)
    plt.ylabel('True positive rate', fontsize=14)
    plt.title('ROC Curve of the PCA-1DCNN Model', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    roc_jpg_path = os.path.join(fig_output_dir, 'PCA-1DCNN_ROC_curve.jpg')
    roc_svg_path = os.path.join(fig_output_dir, 'PCA-1DCNN_ROC_curve.svg')
    plt.savefig(_output(roc_jpg_path), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(_output(roc_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'ROC曲线已保存为：{roc_jpg_path}')
    print(f'ROC曲线已保存为：{roc_svg_path}')
    positive_global_mask = np.zeros(len(y_score_all), dtype=bool)
    positive_global_mask[valid_label_mask_array] = y_true == 1
    total_positive = int(np.sum(positive_global_mask))
    if total_positive == 0:
        raise ValueError('测试集中没有有效正样本，无法绘制P-A plot。')
    score_min = float(np.min(y_score_all))
    score_max = float(np.max(y_score_all))
    if score_min >= 0 and score_max <= 1:
        bins = np.linspace(0, 1, 11, dtype=np.float32)
    else:
        bins = np.linspace(score_min, score_max, 11, dtype=np.float32)
    class_labels = []
    area_percent_by_class = []
    captured_percent_by_class = []
    prediction_efficiency = []
    for i in range(len(bins) - 1):
        lower = float(bins[i])
        upper = float(bins[i + 1])
        if i == len(bins) - 2:
            class_mask = (y_score_all >= lower) & (y_score_all <= upper)
        else:
            class_mask = (y_score_all >= lower) & (y_score_all < upper)
        area_percent = np.sum(class_mask) / len(y_score_all) * 100
        captured_percent = np.sum(positive_global_mask & class_mask) / total_positive * 100
        if area_percent > 0:
            efficiency = captured_percent / area_percent
        else:
            efficiency = np.nan
        class_labels.append(f'{lower:.2f}-{upper:.2f}')
        area_percent_by_class.append(float(area_percent))
        captured_percent_by_class.append(float(captured_percent))
        prediction_efficiency.append(float(efficiency) if np.isfinite(efficiency) else np.nan)
    class_labels = class_labels[::-1]
    area_percent_by_class = area_percent_by_class[::-1]
    captured_percent_by_class = captured_percent_by_class[::-1]
    prediction_efficiency = prediction_efficiency[::-1]
    x_positions = np.arange(len(class_labels))
    bar_width = 0.38
    plt.figure(figsize=(10, 6))
    plt.bar(x_positions - bar_width / 2, area_percent_by_class, bar_width, label='PCA-1DCNN predicted area (%)', edgecolor='black')
    plt.bar(x_positions + bar_width / 2, captured_percent_by_class, bar_width, label='PCA-1DCNN captured positive samples (%)', edgecolor='black')
    plt.xlabel('Prospectivity score class', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.title('Prediction-Area Plot of the PCA-1DCNN Model', fontsize=16)
    plt.xticks(x_positions, class_labels, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y')
    plt.tight_layout()
    pa_jpg_path = os.path.join(fig_output_dir, 'PCA-1DCNN_prediction_area_plot.jpg')
    pa_svg_path = os.path.join(fig_output_dir, 'PCA-1DCNN_prediction_area_plot.svg')
    plt.savefig(_output(pa_jpg_path), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(_output(pa_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'P-A plot已保存为：{pa_jpg_path}')
    print(f'P-A plot已保存为：{pa_svg_path}')
    sorted_indices = np.argsort(y_score_all)[::-1]
    sorted_positive_labels = positive_global_mask[sorted_indices]
    sorted_scores = y_score_all[sorted_indices]
    cumulative_area = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    cumulative_positive_samples = np.cumsum(sorted_positive_labels) / total_positive * 100
    capture_results = []
    for area_threshold in [5, 10, 20, 30, 40, 50]:
        index = np.searchsorted(cumulative_area, area_threshold)
        if index < len(cumulative_positive_samples):
            captured = cumulative_positive_samples[index]
        else:
            captured = 100.0
        capture_results.append([area_threshold, float(captured)])
        print(f'Top {area_threshold}% predicted area captures {captured:.2f}% of positive samples.')
    roc_data_df = pd.DataFrame({'False_positive_rate': fpr, 'True_positive_rate': tpr, 'Threshold': roc_thresholds})
    roc_csv_path = os.path.join(fig_output_dir, 'PCA-1DCNN_ROC_curve_data.csv')
    roc_data_df.to_csv(_output(roc_csv_path), index=False)
    pa_data_df = pd.DataFrame({'Prospectivity_score_class': class_labels, 'Predicted_area_percent': area_percent_by_class, 'Captured_positive_samples_percent': captured_percent_by_class, 'Prediction_efficiency': prediction_efficiency})
    pa_csv_path = os.path.join(fig_output_dir, 'PCA-1DCNN_prediction_area_plot_data.csv')
    pa_data_df.to_csv(_output(pa_csv_path), index=False)
    capture_df = pd.DataFrame(capture_results, columns=['Predicted_area_percent', 'Captured_positive_samples_percent'])
    capture_csv_path = os.path.join(fig_output_dir, 'PCA-1DCNN_capture_results.csv')
    capture_df.to_csv(_output(capture_csv_path), index=False)
    metrics_txt_path = os.path.join(fig_output_dir, 'PCA-1DCNN_model_evaluation_metrics.txt')
    with open(_output(metrics_txt_path), 'w', encoding='utf-8') as file:
        file.write('PCA-1DCNN model evaluation results\n')
        file.write('==================================\n')
        file.write(f'AUC = {roc_auc:.4f}\n')
        file.write(f'Total predicted samples = {len(y_score_all)}\n')
        file.write(f'Total labelled samples = {len(y_true)}\n')
        file.write(f'Positive samples = {np.sum(y_true == 1)}\n')
        file.write(f'Negative samples = {np.sum(y_true == 0)}\n\n')
        file.write('Prediction-area plot results:\n')
        for score_class, area, captured, efficiency in zip(class_labels, area_percent_by_class, captured_percent_by_class, prediction_efficiency):
            file.write(f'{score_class}: predicted area = {area:.2f}%, captured positive samples = {captured:.2f}%, prediction efficiency = {efficiency:.2f}\n')
        file.write('\nPrediction-area capture results:\n')
        for area_threshold, captured in capture_results:
            file.write(f'Top {area_threshold}% predicted area captures {captured:.2f}% of positive samples.\n')
    print(f'预测结果已保存为：{prediction_csv_path}')
    print(f'ROC数据已保存为：{roc_csv_path}')
    print(f'P-A plot数据已保存为：{pa_csv_path}')
    print(f'捕获效率结果已保存为：{capture_csv_path}')
    print(f'模型评价指标已保存为：{metrics_txt_path}')
    print('全部分析完成。')
    export_pa_summary(pa_data_df, 'PCA_1DCNN', 'all prediction pixels; positives are labelled positive pixels')

def original_076():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    from sklearn.metrics import roc_curve, auc
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['mathtext.fontset'] = 'stix'
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/PCA_with_labels_Test.csv'
    model_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/2DCNN_test/ensure'
    model_path = os.path.join(model_dir, '2DCNN_model.h5')
    fig_output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Revised/fig'
    if not os.path.exists(fig_output_dir):
        os.makedirs(str(OUT), exist_ok=True)
    pca_data_test = pd.read_csv(test_file_path)
    print('CSV文件中的列名如下：')
    print(pca_data_test.columns.tolist())
    possible_x_cols = ['x', 'X', 'lon', 'Lon', 'longitude', 'Longitude']
    possible_y_cols = ['y', 'Y', 'lat', 'Lat', 'latitude', 'Latitude']
    x_col = None
    y_col = None
    for col in possible_x_cols:
        if col in pca_data_test.columns:
            x_col = col
            break
    for col in possible_y_cols:
        if col in pca_data_test.columns:
            y_col = col
            break
    if x_col is None or y_col is None:
        raise ValueError('没有找到 x 或 y 坐标列。请检查CSV中的坐标列名，并在 possible_x_cols 或 possible_y_cols 中补充。')
    print(f'识别到的X坐标列为：{x_col}')
    print(f'识别到的Y坐标列为：{y_col}')
    x_coords = pca_data_test[x_col].values
    y_coords = pca_data_test[y_col].values
    possible_label_cols = ['label', 'Label', 'labels', 'Labels', 'class', 'Class', 'target', 'Target', 'deposit', 'Deposit', 'ore', 'Ore', 'positive', 'Positive', 'mineralization', 'Mineralization', '矿床', '标签', '类别']
    label_col = None
    for col in possible_label_cols:
        if col in pca_data_test.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError('没有找到标签列。请检查CSV文件中的真实标签列名，然后把该列名加入 possible_label_cols。')
    print(f'识别到的标签列为：{label_col}')
    label_values = pca_data_test[label_col].values
    label_values = np.asarray(label_values).astype(int)
    candidate_feature_cols = []
    for col in pca_data_test.columns:
        col_lower = str(col).lower()
        if ('pca' in col_lower or col_lower.startswith('pc')) and col not in [x_col, y_col, label_col]:
            candidate_feature_cols.append(col)
    if len(candidate_feature_cols) >= 23:
        feature_cols = candidate_feature_cols[-23:]
    else:
        exclude_cols = [x_col, y_col, label_col]
        feature_df = pca_data_test.drop(columns=exclude_cols)
        feature_df = feature_df.select_dtypes(include=[np.number])
        if feature_df.shape[1] < 23:
            raise ValueError(f'可用数值特征列少于23列，目前只有 {feature_df.shape[1]} 列。')
        feature_cols = feature_df.columns[-23:].tolist()
    print('用于模型输入的23个PCA特征列为：')
    print(feature_cols)
    pca_values = pca_data_test[feature_cols].values
    print(f'PCA feature shape: {pca_values.shape}')
    print(f'Label shape: {label_values.shape}')

    def prepare_test_data_by_coordinates_with_labels(x_coords, y_coords, pca_values, label_values, patch_size=9, min_points=50):
        """
    生成 9 × 9 × 23 的测试patch，并为每个patch生成对应标签。

    如果patch中包含至少一个正样本像元，则该patch标签为1；
    如果patch中不包含正样本像元，则该patch标签为0。
    """
        reshaped_patches = []
        patch_labels = []
        indices = []
        min_x, max_x = (min(x_coords), max(x_coords))
        min_y, max_y = (min(y_coords), max(y_coords))
        x_steps = np.arange(min_x, max_x + 1, patch_size)
        y_steps = np.arange(min_y, max_y + 1, patch_size)
        for y_min in y_steps:
            y_max = y_min + patch_size - 1
            if y_max > max_y:
                continue
            for x_min in x_steps:
                x_max = x_min + patch_size - 1
                if x_max > max_x:
                    continue
                mask = (x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)
                patch_data = pca_values[mask]
                patch_label_data = label_values[mask]
                patch_x = x_coords[mask]
                patch_y = y_coords[mask]
                num_points = len(patch_data)
                if num_points >= min_points:
                    sort_order = np.lexsort((patch_x, patch_y))
                    patch_data = patch_data[sort_order]
                    patch_label_data = patch_label_data[sort_order]
                    patch_label = 1 if np.any(patch_label_data == 1) else 0
                    if num_points < patch_size * patch_size:
                        patch_data = np.pad(patch_data, ((0, patch_size * patch_size - num_points), (0, 0)), mode='constant')
                    try:
                        reshaped_patch = patch_data.reshape(patch_size, patch_size, -1)
                        reshaped_patches.append(reshaped_patch)
                        patch_labels.append(patch_label)
                        indices.append((x_min, x_max, y_min, y_max))
                    except ValueError as e:
                        print(f'Error reshaping patch at X: {x_min}-{x_max}, Y: {y_min}-{y_max} -> {e}')
                else:
                    print(f'Skipping patch at X: {x_min}-{x_max}, Y: {y_min}-{y_max}, insufficient points.')
        print(f'Total patches generated: {len(reshaped_patches)}')
        return (np.array(reshaped_patches), np.array(patch_labels), indices)
    test_patches, y_true, indices = prepare_test_data_by_coordinates_with_labels(x_coords, y_coords, pca_values, label_values, patch_size=9, min_points=50)
    print(f'Test patches shape: {test_patches.shape}')
    print(f'Patch labels shape: {y_true.shape}')
    print(f'Number of positive patches: {np.sum(y_true == 1)}')
    print(f'Number of negative patches: {np.sum(y_true == 0)}')
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        raise ValueError('测试patch中只有一种标签，无法计算ROC曲线和AUC。请检查测试集是否同时包含正样本和负样本。')
    model = load_model(model_path)
    print(f'模型已加载：{model_path}')
    print(f'图片输出目录：{fig_output_dir}')
    predictions = model.predict(test_patches)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        y_score = predictions.ravel()
    elif predictions.ndim == 2 and predictions.shape[1] == 2:
        y_score = predictions[:, 1]
    else:
        y_score = predictions.ravel()
    print(f'Prediction score shape: {y_score.shape}')
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    export_roc_auc('PCA_2DCNN', fpr, tpr, thresholds, roc_auc, y_true, '9x9 patches')
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'PCA-transformed 2DCNN, AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, label='Random prediction')
    plt.xlabel('False positive rate', fontsize=14)
    plt.ylabel('True positive rate', fontsize=14)
    plt.title('ROC Curve of the PCA-transformed 2DCNN Model', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    roc_jpg_path = os.path.join(fig_output_dir, 'PCA-transformed 2DCNN_ROC_curve.jpg')
    roc_svg_path = os.path.join(fig_output_dir, 'PCA-transformed 2DCNN_ROC_curve.svg')
    plt.savefig(_output(roc_jpg_path), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(_output(roc_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'ROC曲线已保存为：{roc_jpg_path}')
    print(f'ROC曲线已保存为：{roc_svg_path}')
    total_positive = np.sum(y_true == 1)
    if total_positive == 0:
        raise ValueError('测试集中没有正样本，无法绘制 P-A plot。')
    score_min = np.min(y_score)
    score_max = np.max(y_score)
    if score_min >= 0 and score_max <= 1:
        bins = np.linspace(0, 1, 11)
    else:
        bins = np.linspace(score_min, score_max, 11)
    class_labels = []
    area_percent_by_class = []
    captured_percent_by_class = []
    prediction_efficiency = []
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        if i == len(bins) - 2:
            class_mask = (y_score >= lower) & (y_score <= upper)
        else:
            class_mask = (y_score >= lower) & (y_score < upper)
        area_percent = np.sum(class_mask) / len(y_score) * 100
        captured_percent = np.sum((y_true == 1) & class_mask) / total_positive * 100
        if area_percent > 0:
            efficiency = captured_percent / area_percent
        else:
            efficiency = np.nan
        class_labels.append(f'{lower:.2f}-{upper:.2f}')
        area_percent_by_class.append(area_percent)
        captured_percent_by_class.append(captured_percent)
        prediction_efficiency.append(efficiency)
    class_labels = class_labels[::-1]
    area_percent_by_class = area_percent_by_class[::-1]
    captured_percent_by_class = captured_percent_by_class[::-1]
    prediction_efficiency = prediction_efficiency[::-1]
    x = np.arange(len(class_labels))
    width = 0.38
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, area_percent_by_class, width, label='PCA-transformed 2DCNN predicted area (%)', edgecolor='black')
    plt.bar(x + width / 2, captured_percent_by_class, width, label='PCA-transformed 2DCNN captured positive samples (%)', edgecolor='black')
    plt.xlabel('Prospectivity score class', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.title('Prediction-Area Plot of the PCA-transformed 2DCNN Model', fontsize=16)
    plt.xticks(x, class_labels, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y')
    plt.tight_layout()
    pa_jpg_path = os.path.join(fig_output_dir, 'PCA-transformed 2DCNN_prediction_area_plot.jpg')
    pa_svg_path = os.path.join(fig_output_dir, 'PCA-transformed 2DCNN_prediction_area_plot.svg')
    plt.savefig(_output(pa_jpg_path), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(_output(pa_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'P-A plot 已保存为：{pa_jpg_path}')
    print(f'P-A plot 已保存为：{pa_svg_path}')
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_labels = y_true[sorted_indices]
    sorted_scores = y_score[sorted_indices]
    cumulative_area = np.arange(1, len(sorted_labels) + 1) / len(sorted_labels) * 100
    cumulative_deposits = np.cumsum(sorted_labels == 1) / total_positive * 100
    capture_results = []
    for area_threshold in [5, 10, 20, 30, 40, 50]:
        idx = np.searchsorted(cumulative_area, area_threshold)
        if idx < len(cumulative_deposits):
            captured = cumulative_deposits[idx]
        else:
            captured = 100
        capture_results.append([area_threshold, captured])
        print(f'Top {area_threshold}% predicted area captures {captured:.2f}% of positive samples.')
    capture_df = pd.DataFrame(capture_results, columns=['Predicted_area_percent', 'Captured_positive_samples_percent'])
    capture_csv_path = os.path.join(fig_output_dir, 'PCA-transformed 2DCNN_capture_results.csv')
    capture_df.to_csv(_output(capture_csv_path), index=False)
    roc_data_df = pd.DataFrame({'False_positive_rate': fpr, 'True_positive_rate': tpr, 'Threshold': thresholds})
    roc_csv_path = os.path.join(fig_output_dir, 'PCA-transformed 2DCNN_ROC_curve_data.csv')
    roc_data_df.to_csv(_output(roc_csv_path), index=False)
    pa_data_df = pd.DataFrame({'Prospectivity_score_class': class_labels, 'Predicted_area_percent': area_percent_by_class, 'Captured_positive_samples_percent': captured_percent_by_class, 'Prediction_efficiency': prediction_efficiency})
    pa_csv_path = os.path.join(fig_output_dir, 'PCA-transformed 2DCNN_prediction_area_plot_data.csv')
    pa_data_df.to_csv(_output(pa_csv_path), index=False)
    metrics_txt_path = os.path.join(fig_output_dir, 'PCA-transformed 2DCNN_model_evaluation_metrics.txt')
    with open(_output(metrics_txt_path), 'w', encoding='utf-8') as f:
        f.write('PCA-transformed 2DCNN model evaluation results\n')
        f.write('=============================================\n')
        f.write(f'AUC = {roc_auc:.4f}\n')
        f.write(f'Total patches = {len(y_true)}\n')
        f.write(f'Positive patches = {np.sum(y_true == 1)}\n')
        f.write(f'Negative patches = {np.sum(y_true == 0)}\n\n')
        f.write('Prediction-area plot results:\n')
        for label, area, captured, efficiency in zip(class_labels, area_percent_by_class, captured_percent_by_class, prediction_efficiency):
            f.write(f'{label}: predicted area = {area:.2f}%, captured positive samples = {captured:.2f}%, prediction efficiency = {efficiency:.2f}\n')
        f.write('\nPrediction-area capture results:\n')
        for area_threshold, captured in capture_results:
            f.write(f'Top {area_threshold}% predicted area captures {captured:.2f}% of positive samples.\n')
    print(f'捕获效率结果已保存为：{capture_csv_path}')
    print(f'ROC数据已保存为：{roc_csv_path}')
    print(f'P-A plot 数据已保存为：{pa_csv_path}')
    print(f'模型评价指标已保存为：{metrics_txt_path}')
    print('全部分析完成。')
    export_pa_summary(pa_data_df, 'PCA_2DCNN', 'nonoverlapping 9x9 patches; any positive pixel makes positive patch')

def original_077():
    """
Evaluate the PCA-RF model using ROC curve and prediction-area plot.

This script follows the logic of PCA_CNN_RF_test.py:
1. Load PCA training and test data
2. Train Random Forest using fixed best parameters
3. Use the trained best_model to predict probabilities
4. Plot ROC curve and Prediction-Area plot

All outputs are saved to:
D:\\homework\\夏季学期\\地化大数据\\central asian mapping\\Revised\\fig

Output figures:
1. PCA-RF_ROC_curve.jpg / .svg
2. PCA-RF_prediction_area_plot.jpg / .svg

All figure fonts are set to Times New Roman.
"""
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve, auc
    import joblib
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['mathtext.fontset'] = 'stix'
    train_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/PCA_with_labels_Train.csv'
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/PCA_with_labels_Test.csv'
    fig_output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Revised/fig'
    if not os.path.exists(fig_output_dir):
        os.makedirs(str(OUT), exist_ok=True)
    feature_cols = [f'PCA{i}' for i in range(1, 24)]
    label_col = 'Label'
    train_header = pd.read_csv(train_file_path, nrows=0).columns.tolist()
    test_header = pd.read_csv(test_file_path, nrows=0).columns.tolist()
    print('训练集CSV文件中的列名如下：')
    print(train_header)
    print('测试集CSV文件中的列名如下：')
    print(test_header)
    missing_train_cols = [col for col in feature_cols + [label_col] if col not in train_header]
    missing_test_cols = [col for col in feature_cols + [label_col] if col not in test_header]
    if len(missing_train_cols) > 0:
        raise ValueError(f'训练集中缺少以下列：{missing_train_cols}')
    if len(missing_test_cols) > 0:
        raise ValueError(f'测试集中缺少以下列：{missing_test_cols}')
    possible_x_cols = ['x', 'X', 'lon', 'Lon', 'longitude', 'Longitude']
    possible_y_cols = ['y', 'Y', 'lat', 'Lat', 'latitude', 'Latitude']
    x_col = None
    y_col = None
    for col in possible_x_cols:
        if col in test_header:
            x_col = col
            break
    for col in possible_y_cols:
        if col in test_header:
            y_col = col
            break
    if x_col is not None:
        print(f'识别到的X坐标列为：{x_col}')
    else:
        print('未识别到X坐标列。')
    if y_col is not None:
        print(f'识别到的Y坐标列为：{y_col}')
    else:
        print('未识别到Y坐标列。')
    train_usecols = feature_cols + [label_col]
    test_usecols = feature_cols + [label_col]
    if x_col is not None:
        test_usecols.append(x_col)
    if y_col is not None:
        test_usecols.append(y_col)
    train_usecols = list(dict.fromkeys(train_usecols))
    test_usecols = list(dict.fromkeys(test_usecols))
    dtype_train = {col: np.float32 for col in feature_cols}
    dtype_train[label_col] = np.float32
    dtype_test = {col: np.float32 for col in feature_cols}
    dtype_test[label_col] = np.float32
    if x_col is not None:
        dtype_test[x_col] = np.float32
    if y_col is not None:
        dtype_test[y_col] = np.float32
    print('正在读取训练集...')
    pca_data_train = pd.read_csv(train_file_path, usecols=train_usecols, dtype=dtype_train)
    print('正在读取测试集...')
    pca_data_test = pd.read_csv(test_file_path, usecols=test_usecols, dtype=dtype_test)
    print(f'训练集样本数：{len(pca_data_train)}')
    print(f'测试集样本数：{len(pca_data_test)}')
    X_train_all = pca_data_train[feature_cols].to_numpy(dtype=np.float32, copy=True)
    train_label_numeric = pd.to_numeric(pca_data_train[label_col], errors='coerce')
    valid_train_mask = train_label_numeric.notna() & train_label_numeric.isin([0, 1])
    X_train = X_train_all[valid_train_mask.to_numpy()]
    y_train = train_label_numeric.loc[valid_train_mask].astype(np.int8).to_numpy()
    print(f'用于训练的样本数：{len(y_train)}')
    print(f'训练集正样本数：{np.sum(y_train == 1)}')
    print(f'训练集负样本数：{np.sum(y_train == 0)}')
    if len(np.unique(y_train)) < 2:
        raise ValueError('训练集中只有一种标签，无法训练二分类 Random Forest 模型。')
    if not np.isfinite(X_train).all():
        print('训练特征中存在 NaN 或无穷值，已替换为 0。')
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_test = pca_data_test[feature_cols].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(X_test).all():
        print('测试特征中存在 NaN 或无穷值，已替换为 0。')
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    best_params = {'n_estimators': 248, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_depth': 9, 'bootstrap': False}
    print('PCA-RF 使用的固定最佳参数为：')
    print(best_params)
    best_model = RandomForestClassifier(**best_params, random_state=0)
    best_model.fit(X_train, y_train)
    print('PCA-RF 模型训练完成。')
    rf_model_path = os.path.join(fig_output_dir, 'PCA-RF_trained_model.joblib')
    joblib.dump(best_model, _output(rf_model_path))
    best_params_txt_path = os.path.join(fig_output_dir, 'PCA-RF_best_params.txt')
    with open(_output(best_params_txt_path), 'w', encoding='utf-8') as f:
        for param, value in best_params.items():
            f.write(f'{param}: {value}\n')
    print(f'PCA-RF模型已保存为：{rf_model_path}')
    print(f'PCA-RF最佳参数已保存为：{best_params_txt_path}')
    y_score_all = best_model.predict_proba(X_test)[:, 1]
    y_score_all = y_score_all.astype(np.float32)
    print(f'预测概率数量：{len(y_score_all)}')
    print(f'最低预测概率：{np.min(y_score_all):.6f}')
    print(f'最高预测概率：{np.max(y_score_all):.6f}')
    print(f'预测概率唯一值数量：{len(np.unique(np.round(y_score_all, 4)))}')
    print(f'预测概率等于0的数量：{np.sum(y_score_all == 0)}')
    print(f'预测概率等于1的数量：{np.sum(y_score_all == 1)}')
    prediction_output = {}
    if x_col is not None:
        prediction_output[x_col] = pca_data_test[x_col].to_numpy()
    if y_col is not None:
        prediction_output[y_col] = pca_data_test[y_col].to_numpy()
    prediction_output[label_col] = pca_data_test[label_col].to_numpy()
    prediction_output['PCA-RF_Predicted_Probability'] = y_score_all
    prediction_df = pd.DataFrame(prediction_output)
    prediction_csv_path = os.path.join(fig_output_dir, 'PCA-RF_predictions.csv')
    prediction_df.to_csv(_output(prediction_csv_path), index=False)
    print(f'预测结果已保存为：{prediction_csv_path}')
    test_label_numeric = pd.to_numeric(pca_data_test[label_col], errors='coerce')
    valid_label_mask = test_label_numeric.notna() & test_label_numeric.isin([0, 1])
    valid_label_mask_array = valid_label_mask.to_numpy()
    print(f'测试集总样本数：{len(pca_data_test)}')
    print(f'有效0/1标签样本数：{np.sum(valid_label_mask_array)}')
    print(f'空标签或非0/1标签样本数：{len(pca_data_test) - np.sum(valid_label_mask_array)}')
    if np.sum(valid_label_mask_array) == 0:
        raise ValueError('测试集中没有有效的0/1标签，无法计算ROC曲线和P-A plot。')
    y_true = test_label_numeric.loc[valid_label_mask].astype(np.int8).to_numpy()
    y_score_roc = y_score_all[valid_label_mask_array]
    print(f'用于ROC评价的样本数：{len(y_true)}')
    print(f'测试集正样本数：{np.sum(y_true == 1)}')
    print(f'测试集负样本数：{np.sum(y_true == 0)}')
    if len(np.unique(y_true)) < 2:
        raise ValueError('有效标签样本中只有一种类别，无法计算ROC曲线和AUC。')
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score_roc)
    roc_auc = auc(fpr, tpr)
    export_roc_auc('PCA_RF', fpr, tpr, roc_thresholds, roc_auc, y_true, 'labelled pixels')
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'PCA-RF, AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, label='Random prediction')
    plt.xlabel('False positive rate', fontsize=14)
    plt.ylabel('True positive rate', fontsize=14)
    plt.title('ROC Curve of the PCA-RF Model', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    roc_jpg_path = os.path.join(fig_output_dir, 'PCA-RF_ROC_curve.jpg')
    roc_svg_path = os.path.join(fig_output_dir, 'PCA-RF_ROC_curve.svg')
    plt.savefig(_output(roc_jpg_path), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(_output(roc_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'ROC曲线已保存为：{roc_jpg_path}')
    print(f'ROC曲线已保存为：{roc_svg_path}')
    positive_global_mask = np.zeros(len(y_score_all), dtype=bool)
    positive_global_mask[valid_label_mask_array] = y_true == 1
    total_positive = int(np.sum(positive_global_mask))
    if total_positive == 0:
        raise ValueError('测试集中没有有效正样本，无法绘制 P-A plot。')
    score_min = float(np.min(y_score_all))
    score_max = float(np.max(y_score_all))
    if score_min >= 0 and score_max <= 1:
        bins = np.linspace(0, 1, 11)
    else:
        bins = np.linspace(score_min, score_max, 11)
    class_labels = []
    area_percent_by_class = []
    captured_percent_by_class = []
    prediction_efficiency = []
    sample_count_by_class = []
    positive_count_by_class = []
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        if i == len(bins) - 2:
            class_mask = (y_score_all >= lower) & (y_score_all <= upper)
        else:
            class_mask = (y_score_all >= lower) & (y_score_all < upper)
        sample_count = np.sum(class_mask)
        positive_count = np.sum(positive_global_mask & class_mask)
        area_percent = sample_count / len(y_score_all) * 100
        captured_percent = positive_count / total_positive * 100
        if area_percent > 0:
            efficiency = captured_percent / area_percent
        else:
            efficiency = np.nan
        class_labels.append(f'{lower:.2f}-{upper:.2f}')
        area_percent_by_class.append(float(area_percent))
        captured_percent_by_class.append(float(captured_percent))
        prediction_efficiency.append(float(efficiency) if np.isfinite(efficiency) else np.nan)
        sample_count_by_class.append(int(sample_count))
        positive_count_by_class.append(int(positive_count))
    class_labels = class_labels[::-1]
    area_percent_by_class = area_percent_by_class[::-1]
    captured_percent_by_class = captured_percent_by_class[::-1]
    prediction_efficiency = prediction_efficiency[::-1]
    sample_count_by_class = sample_count_by_class[::-1]
    positive_count_by_class = positive_count_by_class[::-1]
    x_positions = np.arange(len(class_labels))
    bar_width = 0.38
    plt.figure(figsize=(10, 6))
    plt.bar(x_positions - bar_width / 2, area_percent_by_class, bar_width, label='PCA-RF predicted area (%)', edgecolor='black')
    plt.bar(x_positions + bar_width / 2, captured_percent_by_class, bar_width, label='PCA-RF captured positive samples (%)', edgecolor='black')
    plt.xlabel('Prospectivity score class', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.title('Prediction-Area Plot of the PCA-RF Model', fontsize=16)
    plt.xticks(x_positions, class_labels, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y')
    plt.tight_layout()
    pa_jpg_path = os.path.join(fig_output_dir, 'PCA-RF_prediction_area_plot.jpg')
    pa_svg_path = os.path.join(fig_output_dir, 'PCA-RF_prediction_area_plot.svg')
    plt.savefig(_output(pa_jpg_path), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(_output(pa_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'P-A plot已保存为：{pa_jpg_path}')
    print(f'P-A plot已保存为：{pa_svg_path}')
    sorted_indices = np.argsort(y_score_all)[::-1]
    sorted_positive_labels = positive_global_mask[sorted_indices]
    sorted_scores = y_score_all[sorted_indices]
    cumulative_area = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    cumulative_positive_samples = np.cumsum(sorted_positive_labels) / total_positive * 100
    capture_results = []
    for area_threshold in [5, 10, 20, 30, 40, 50]:
        idx = np.searchsorted(cumulative_area, area_threshold)
        if idx < len(cumulative_positive_samples):
            captured = cumulative_positive_samples[idx]
        else:
            captured = 100.0
        capture_results.append([area_threshold, captured])
        print(f'Top {area_threshold}% predicted area captures {captured:.2f}% of positive samples.')
    roc_data_df = pd.DataFrame({'False_positive_rate': fpr, 'True_positive_rate': tpr, 'Threshold': roc_thresholds})
    roc_csv_path = os.path.join(fig_output_dir, 'PCA-RF_ROC_curve_data.csv')
    roc_data_df.to_csv(_output(roc_csv_path), index=False)
    pa_data_df = pd.DataFrame({'Prospectivity_score_class': class_labels, 'Sample_count': sample_count_by_class, 'Positive_sample_count': positive_count_by_class, 'Predicted_area_percent': area_percent_by_class, 'Captured_positive_samples_percent': captured_percent_by_class, 'Prediction_efficiency': prediction_efficiency})
    pa_csv_path = os.path.join(fig_output_dir, 'PCA-RF_prediction_area_plot_data.csv')
    pa_data_df.to_csv(_output(pa_csv_path), index=False)
    capture_df = pd.DataFrame(capture_results, columns=['Predicted_area_percent', 'Captured_positive_samples_percent'])
    capture_csv_path = os.path.join(fig_output_dir, 'PCA-RF_capture_results.csv')
    capture_df.to_csv(_output(capture_csv_path), index=False)
    metrics_txt_path = os.path.join(fig_output_dir, 'PCA-RF_model_evaluation_metrics.txt')
    with open(_output(metrics_txt_path), 'w', encoding='utf-8') as f:
        f.write('PCA-RF model evaluation results\n')
        f.write('===============================\n')
        f.write(f'AUC = {roc_auc:.4f}\n')
        f.write(f'Total predicted samples = {len(y_score_all)}\n')
        f.write(f'Total labelled samples = {len(y_true)}\n')
        f.write(f'Positive samples = {np.sum(y_true == 1)}\n')
        f.write(f'Negative samples = {np.sum(y_true == 0)}\n')
        f.write(f'Min predicted probability = {np.min(y_score_all):.6f}\n')
        f.write(f'Max predicted probability = {np.max(y_score_all):.6f}\n')
        f.write(f'Unique predicted probabilities, rounded to 4 decimals = {len(np.unique(np.round(y_score_all, 4)))}\n')
        f.write(f'Predicted probabilities equal to 0 = {np.sum(y_score_all == 0)}\n')
        f.write(f'Predicted probabilities equal to 1 = {np.sum(y_score_all == 1)}\n\n')
        f.write('Prediction-area plot results:\n')
        for score_class, sample_count, positive_count, area, captured, efficiency in zip(class_labels, sample_count_by_class, positive_count_by_class, area_percent_by_class, captured_percent_by_class, prediction_efficiency):
            f.write(f'{score_class}: sample count = {sample_count}, positive sample count = {positive_count}, predicted area = {area:.2f}%, captured positive samples = {captured:.2f}%, prediction efficiency = {efficiency:.2f}\n')
        f.write('\nPrediction-area capture results:\n')
        for area_threshold, captured in capture_results:
            f.write(f'Top {area_threshold}% predicted area captures {captured:.2f}% of positive samples.\n')
    feature_contributions = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Gini_importance': feature_contributions})
    feature_importance_csv_path = os.path.join(fig_output_dir, 'PCA-RF_feature_importance.csv')
    feature_importance_df.to_csv(_output(feature_importance_csv_path), index=False)
    print(f'预测结果已保存为：{prediction_csv_path}')
    print(f'ROC数据已保存为：{roc_csv_path}')
    print(f'P-A plot数据已保存为：{pa_csv_path}')
    print(f'捕获效率结果已保存为：{capture_csv_path}')
    print(f'模型评价指标已保存为：{metrics_txt_path}')
    print(f'特征重要性结果已保存为：{feature_importance_csv_path}')
    print('全部分析完成。')
    export_pa_summary(pa_data_df, 'PCA_RF', 'all prediction pixels; positives are labelled positive pixels')

def original_079():
    """
使用最佳参数的随机森林模型进行训练，计算各PCA特征的贡献度 (基于 Gini 系数)，并绘制柱状图
"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from sklearn.ensemble import RandomForestClassifier
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/RF_test'
    os.makedirs(str(OUT), exist_ok=True)
    train_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/PCA_with_labels_Train.csv'
    test_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/PCA_with_labels_Test.csv'
    pca_data_train = pd.read_csv(train_file_path)
    pca_data_test = pd.read_csv(test_file_path)
    X_train = pca_data_train.loc[:, 'PCA1':'PCA23'].values
    y_train = pca_data_train['Label'].values
    X_test = pca_data_test.loc[:, 'PCA1':'PCA23'].values
    best_params = {'n_estimators': 248, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_depth': 9, 'bootstrap': False}
    best_model = RandomForestClassifier(**best_params, random_state=0)
    best_model.fit(X_train, y_train)
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/RF_test'
    os.makedirs(str(OUT), exist_ok=True)
    predicted_values_file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/RF_test/predicted_values_RF_test.csv'
    old_prediction = pd.read_csv(predicted_values_file_path)
    predicted_data = pca_data_test.copy()
    predicted_data['Predicted_Probability'] = best_model.predict_proba(X_test)[:, 1]
    predicted_data.to_csv(OUT / 'RF_REFIT__full_predictions.csv', index=False)
    compare_coordinate_values(predicted_data.copy(), old_prediction, ['Predicted_Probability'], 'RF_REFIT__vs_saved_predictions')
    if {'x', 'y', 'Predicted_Probability'}.issubset(predicted_data.columns):
        x_coords = predicted_data['x'].values
        y_coords = predicted_data['y'].values
        predicted_probs = predicted_data['Predicted_Probability'].values
        valid_indices = ~np.isnan(predicted_probs)
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]
        predicted_probs = predicted_probs[valid_indices]
        plt.figure(figsize=(18, 5.64), dpi=100)
        plt.tricontourf(x_coords, y_coords, predicted_probs, levels=14, cmap='coolwarm')
        plt.colorbar(label='Predicted Probability')
        plt.axis('off')
        plt.title('Predicted Probability Contour Map - RandomForest (predicted_values_RF_test)')
        plt.savefig(_output(os.path.join(output_dir, 'contour_map_predicted_values_RF_test.png')), dpi=100, bbox_inches='tight')
        plt.savefig(_output(os.path.join(output_dir, 'contour_map_predicted_values_RF_test.svg')), format='svg', bbox_inches='tight')
        plt.show()
    else:
        print("文件中未找到 'x', 'y' 或 'Predicted_Probability' 列。")
    feature_contributions = best_model.feature_importances_
    for idx, contribution in enumerate(feature_contributions, start=1):
        print(f'PCA{idx} 贡献度: {contribution:.4f}')
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/RF_test'
    os.makedirs(str(OUT), exist_ok=True)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(1, 24), feature_contributions, color='skyblue', edgecolor='black')
    plt.xticks(ticks=range(1, 24))
    plt.xlabel('PCA Feature Index', fontsize=12)
    plt.ylabel('Average Contribution to Prediction (Gini Importance)', fontsize=12)
    plt.title('PCA Feature Sensitivity Analysis', fontsize=14)
    output_png_path = os.path.join(output_dir, 'pca_feature_contributions_RF_test.png')
    output_svg_path = os.path.join(output_dir, 'pca_feature_contributions_RF_test.svg')
    plt.savefig(_output(output_png_path), format='png', dpi=100, bbox_inches='tight')
    plt.savefig(_output(output_svg_path), format='svg', bbox_inches='tight')
    plt.show()
    print(f'Feature contribution plots saved as: {output_png_path} and {output_svg_path}')
    export_prediction_map(predicted_data, predicted_data['Predicted_Probability'].to_numpy(), 'PCA_RF_REFIT')
    pd.DataFrame({'PCA': [f'PCA{i + 1}' for i in range(23)], 'Gini_importance': feature_contributions}).to_csv(OUT / 'PCA_RF_REFIT__Gini_importance.csv', index=False)

def original_080():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    label_data_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/data/Sample Location_delet_orogenic_gold.xlsx'
    df = pd.read_excel(label_data_path)
    labels = np.full((564, 1800), np.nan)
    longitude_min, longitude_max = (55, 135)
    latitude_min, latitude_max = (35, 60)
    x_pixels = (df['Longitude'] - longitude_min) / (longitude_max - longitude_min) * 1799
    y_pixels = (df['Latitude'] - latitude_min) / (latitude_max - latitude_min) * 563
    fig, ax = plt.subplots(figsize=(283.464 / 25.4, 110.307 / 25.4))
    ax.scatter(x_pixels, y_pixels, c='gold', s=200, edgecolors='black', marker='o')
    ax.set_xlim(0, 1799)
    ax.set_ylim(0, 563)
    ax.set_title('矿点位置图', fontsize=16)
    ax.set_xlabel('X 像素坐标')
    ax.set_ylabel('Y 像素坐标')
    plt.savefig(_output('output_image.jpg'), format='jpg', dpi=300)
    plt.savefig(_output('output_image.svg'), format='svg')
    plt.show()

def original_082():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Th_U_and_temperature/Magamatic_Th_U_and_T.xlsx'
    df = pd.read_excel(file_path)
    temperature = df['Temperature']
    th_u = df['Th/U']
    zircon_zone = df['zircon zone']

    def get_rgb(temp):
        """根据温度值返回相应的RGB颜色"""
        if temp < 650:
            return (64 / 255, 39 / 255, 169 / 255)
        elif 650 <= temp < 700:
            return (66 / 255, 108 / 255, 255 / 255)
        elif 700 <= temp < 750:
            return (3 / 255, 185 / 255, 196 / 255)
        elif 750 <= temp < 800:
            return (69 / 255, 204 / 255, 83 / 255)
        elif 800 <= temp < 850:
            return (255 / 255, 190 / 255, 62 / 255)
        elif 850 <= temp < 900:
            return (245 / 255, 227 / 255, 41 / 255)
        else:
            return (250 / 255, 252 / 255, 21 / 255)

    def get_marker(zone):
        """根据'zircon zone'列的值返回不同的标记"""
        if zone == 'i':
            return 's'
        elif zone == 'c':
            return 'o'
        elif zone == 'r':
            return 'D'
        else:
            return 'o'
    df_less_714 = df[df['Temperature'] < 714]
    df_greater_714 = df[df['Temperature'] >= 714]
    plt.figure(figsize=(16, 8))
    for i in range(len(df)):
        color = get_rgb(temperature[i])
        marker = get_marker(zircon_zone[i])
        plt.scatter(temperature[i], th_u[i], color=color, marker=marker, label=zircon_zone[i] if i == 0 else '', s=150, edgecolor='black', linewidth=0.5)
    coeffs_less_714 = np.polyfit(df_less_714['Temperature'], df_less_714['Th/U'], deg=1)
    poly_less_714 = np.poly1d(coeffs_less_714)
    coeffs_greater_714 = np.polyfit(df_greater_714['Temperature'], df_greater_714['Th/U'], deg=1)
    poly_greater_714 = np.poly1d(coeffs_greater_714)
    temp_range_1 = np.linspace(df_less_714['Temperature'].min(), df_less_714['Temperature'].max(), 100)
    temp_range_2 = np.linspace(df_greater_714['Temperature'].min(), df_greater_714['Temperature'].max(), 100)
    plt.plot(temp_range_1, poly_less_714(temp_range_1), color='blue', label=f'Fit (T < 714): y = {coeffs_less_714[0]:.3e}x + {coeffs_less_714[1]:.3e}')
    plt.plot(temp_range_2, poly_greater_714(temp_range_2), color='green', label=f'Fit (T >= 714): y = {coeffs_greater_714[0]:.3e}x + {coeffs_greater_714[1]:.3e}')
    fit_results = {'Temperature Range': ['T < 714', 'T >= 714'], 'a (slope)': [coeffs_less_714[0], coeffs_greater_714[0]], 'b (intercept)': [coeffs_less_714[1], coeffs_greater_714[1]], 'Formula': [f'y = {coeffs_less_714[0]:.3e}x + {coeffs_less_714[1]:.3e}', f'y = {coeffs_greater_714[0]:.3e}x + {coeffs_greater_714[1]:.3e}']}
    fit_results_df = pd.DataFrame(fit_results)
    output_excel_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Th_U_and_temperature/fit_results_linear.xlsx'
    fit_results_df.to_excel(_output(output_excel_path), index=False)
    plt.yscale('log')
    plt.xlim(600, 900)
    plt.ylim(0.15, 2)
    plt.axvline(x=856, color='black', linestyle='--', label='856°C')
    plt.axvline(x=722, color='black', linestyle='--', label='722°C')
    plt.axvline(x=714, color='black', linestyle='--', label='714°C')
    plt.legend()
    plt.title('Th/U vs Temperature with Zircon Zone')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Th/U')
    output_dir = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/Th_U_and_temperature'
    plt.savefig(_output(f'{output_dir}\\scatter_plot_with_linear_fit.png'), format='png')
    plt.savefig(_output(f'{output_dir}\\scatter_plot_with_linear_fit.svg'), format='svg')
    plt.show()

def original_083():
    import pandas as pd
    df = pd.read_excel('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/U-Pb_Best_age/Detrital zircon U-Pb age--坐标处理_数据处理_Delet_Best_age.xlsx')

    def calculate_discordance(row):
        if row['Published 206Pb/238 U age(Ma)'] <= 1200:
            discordance = (row['Published 206Pb/238 U age(Ma)'] - row['Published 207Pb/235U age(Ma)']) / row['Published 206Pb/238 U age(Ma)'] * 100
        else:
            discordance = (row['Published 207Pb/206 Pb age(Ma)'] - row['Published 206Pb/238 U age(Ma)']) / row['Published 207Pb/206 Pb age(Ma)'] * 100
        return discordance
    df['Discordance'] = df.apply(calculate_discordance, axis=1)

    def calculate_best_age(row):
        if abs(row['Discordance']) > 30:
            return None
        if row['Published 206Pb/238 U age(Ma)'] <= 1200:
            return row['Published 206Pb/238 U age(Ma)']
        else:
            return row['Published 207Pb/206 Pb age(Ma)']
    df['Best age'] = df.apply(calculate_best_age, axis=1)
    df = df.dropna(subset=['Best age'])

    def assign_age_label(age):
        if age <= 66:
            return 'Cenozoic'
        elif 66 < age <= 251.902:
            return 'Mesozoic'
        elif 251.902 < age <= 538.8:
            return 'Paleozoic'
        elif 538.8 < age <= 4567:
            return 'Precambrian'
        else:
            return 'Unknown'
    df['Era'] = df['Best age'].apply(assign_age_label)
    df.to_excel(_output('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/U-Pb_Best_age/output_with_best_age_and_era_labels.xlsx'), index=False)
    print('最佳年龄计算完成，并已为每个样品打上相应的时代标签。')

def original_084():
    """
Created on Fri Oct 18 20:44:26 2024

@author: ASUS
"""
    import pandas as pd
    df = pd.read_excel('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/U-Pb_Best_age/Detrital zircon U-Pb age--坐标处理_数据处理_Delet_Best_age.xlsx')

    def calculate_discordance(row):
        if row['Published 206Pb/238 U age(Ma)'] <= 1200:
            discordance = (row['Published 206Pb/238 U age(Ma)'] - row['Published 207Pb/235U age(Ma)']) / row['Published 206Pb/238 U age(Ma)'] * 100
        else:
            discordance = (row['Published 207Pb/206 Pb age(Ma)'] - row['Published 206Pb/238 U age(Ma)']) / row['Published 207Pb/206 Pb age(Ma)'] * 100
        return discordance
    df['Discordance'] = df.apply(calculate_discordance, axis=1)

    def calculate_best_age(row):
        if abs(row['Discordance']) > 30:
            return None
        if row['Published 206Pb/238 U age(Ma)'] <= 1200:
            return row['Published 206Pb/238 U age(Ma)']
        else:
            return row['Published 207Pb/206 Pb age(Ma)']
    df['Best age'] = df.apply(calculate_best_age, axis=1)
    df = df.dropna(subset=['Best age'])

    def assign_period_label(age):
        if age <= 66:
            return 'Cenozoic'
        elif 66 < age <= 251.902:
            return 'Mesozoic'
        elif 251.902 < age <= 538.8:
            return 'Paleozoic'
        elif 538.8 < age <= 1000:
            return 'Neoproterozoic'
        elif 1000 < age <= 1600:
            return 'Mesoproterozoic'
        elif 1600 < age <= 2500:
            return 'Paleoproterozoic'
        elif 2500 < age <= 4031:
            return 'Archean'
        elif 4031 < age <= 4567:
            return 'Hadean'
        else:
            return None
    df['Period'] = df['Best age'].apply(assign_period_label)
    df = df.dropna(subset=['Period'])
    df.to_excel(_output('D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/U-Pb_Best_age/output_with_best_age_and_period_labels.xlsx'), index=False)
    print('最佳年龄计算完成，并已为每个样品打上相应的时期标签，超过4567 Ma的数据已删除。')

def original_085():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    file_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/U-Pb_Best_age/output_with_best_age_and_period_labels.xlsx'
    df = pd.read_excel(file_path)
    era_order = ['Cenozoic', 'Mesozoic', 'Paleozoic', 'Precambrian']
    max_age = 4567
    plt.figure(figsize=(10, 6))
    for era in era_order:
        era_data = df[df['Era'] == era]['Best age'].dropna()
        sorted_data = np.sort(era_data)
        sorted_data = np.insert(sorted_data, 0, 0)
        if sorted_data[-1] < max_age:
            sorted_data = np.append(sorted_data, max_age)
        yvals = np.arange(len(sorted_data)) / (len(sorted_data) - 1)
        if yvals[-1] != 1:
            yvals = np.append(yvals, 1)
            sorted_data = np.append(sorted_data, max_age)
        plt.plot(sorted_data, yvals, label=era)
    total_data = df['Best age'].dropna()
    sorted_total_data = np.sort(total_data)
    sorted_total_data = np.insert(sorted_total_data, 0, 0)
    sorted_total_data = np.append(sorted_total_data, max_age)
    yvals_total = np.arange(len(sorted_total_data)) / (len(sorted_total_data) - 1)
    if yvals_total[-1] != 1:
        yvals_total = np.append(yvals_total, 1)
        sorted_total_data = np.append(sorted_total_data, max_age)
    plt.plot(sorted_total_data, yvals_total, label='Total', color='black', linestyle='--')
    plt.legend()
    plt.title('Cumulative Distribution')
    plt.xlabel('Best age (Ma)')
    plt.ylabel('Cumulative Distribution')
    plt.ylim(0, 1)
    plt.xlim(0, max_age)
    plt.xticks(np.arange(0, max_age + 1, 500))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    output_png_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/U-Pb_Best_age/cumulative_distribution_plot_fixed.png'
    output_svg_path = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/U-Pb_Best_age/cumulative_distribution_plot_fixed.svg'
    plt.savefig(_output(output_png_path), format='png', dpi=300)
    plt.savefig(_output(output_svg_path), format='svg')
    plt.show()

def original_089():
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    import os
    import glob
    import numpy as np
    import pandas as pd
    from scipy.stats import normaltest, anderson
    from scipy.spatial.distance import pdist
    from pysal.explore import esda
    from pysal.lib import weights
    folder = 'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data/相关性分析'
    output_dir = os.path.join(folder, 'Output')
    os.makedirs(str(OUT), exist_ok=True)
    excel_files = [f for f in glob.glob(os.path.join(folder, '*.xlsx')) if not os.path.basename(f).startswith('~$')]
    results = []
    for file_path in excel_files:
        name = os.path.splitext(os.path.basename(file_path))[0]
        print(f'开始处理：{name}')
        df = pd.read_excel(file_path, header=0)
        y_raw = df.iloc[:, 0].astype(str)
        x_raw = df.iloc[:, 1].astype(str)
        vals = df.iloc[:, 2].values
        y_num = pd.to_numeric(y_raw.str.extract('(\\d+)')[0], errors='coerce')
        x_num = pd.to_numeric(x_raw.str.extract('(\\d+)')[0], errors='coerce')
        mask = ~np.isnan(vals) & ~y_num.isna() & ~x_num.isna()
        coords = np.column_stack((y_num[mask], x_num[mask]))
        data_array = vals[mask]
        n_points = data_array.size
        if n_points >= 8:
            try:
                dag_stat, dag_p = normaltest(data_array)
            except:
                dag_stat, dag_p = (np.nan, np.nan)
        else:
            dag_stat, dag_p = (np.nan, np.nan)
        if n_points >= 8:
            try:
                ad_res = anderson(data_array, dist='norm')
                sl, cv = (ad_res.significance_level, ad_res.critical_values)
                try:
                    idx5 = list(sl).index(5.0)
                except ValueError:
                    idx5 = int(np.argmin(np.abs(np.array(sl) - 5.0)))
                ad_stat, ad_crit5 = (ad_res.statistic, cv[idx5])
            except:
                ad_stat, ad_crit5 = (np.nan, np.nan)
        else:
            ad_stat, ad_crit5 = (np.nan, np.nan)
        if coords.shape[0] > 1:
            try:
                k = min(6, coords.shape[0] - 1)
                if coords.shape[0] > k:
                    w = weights.KNN.from_array(coords, k=k)
                else:
                    threshold = np.median(pdist(coords))
                    w = weights.DistanceBand(coords, threshold=threshold, binary=True)
                moran = esda.Moran(data_array, w)
                mi, mp = (moran.I, moran.p_sim)
            except:
                mi, mp = (np.nan, np.nan)
        else:
            mi, mp = (np.nan, np.nan)
        print(f'  {name}：D’Agostino p={dag_p:.2e}, Anderson stat={ad_stat:.2f}, crit5%={ad_crit5:.2f}, Moran’s I={mi:.4f}, p={mp:.4f}')
        results.append({'filename': name, 'dagostino_stat': dag_stat, 'dagostino_pvalue': dag_p, 'anderson_statistic': ad_stat, 'anderson_crit_5%': ad_crit5, 'moran_I': mi, 'moran_pvalue': mp})
    result_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'correlation_analysis_results.xlsx')
    result_df.to_excel(_output(output_path), index=False)
    print(f'全部完成，结果已保存至：{output_path}')

def run_stage(item):
    global ACTIVE_STAGE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ACTIVE_STAGE = item['function']
    previous = Path.cwd()
    plt.close('all')
    plt.rcdefaults()
    configure_chinese_fonts()
    original_show = plt.show
    plt.show = lambda *a, **k: plt.close('all')
    try:
        os.chdir(DATA_ROOT)
        globals()[item['function']]()
    finally:
        plt.close('all')
        plt.show = original_show
        os.chdir(previous)
        gc.collect()
CATALOG = [{'path': 'sample location.py', 'sha256': 'e7f81d7d0e09ed11b758d1471ea545e692cc1911c671a53e080f4e3600e56d1d', 'function': 'original_080', 'default': True, 'branch': 'reproduce'}, {'path': 'original_analyses/original_2DCNN_test_frozen.py', 'sha256': '5710f693779e26f32e08f3e1bb9bf7010a5dbbf09dc2373a2a6f22306ff75dd7', 'function': 'original_056', 'default': True, 'branch': 'reproduce'}, {'path': '1DCNN_test/PCA_CNN_1DCNN_test_fraze.py', 'sha256': 'adb55f92ba7316f589422f94ec58b629a55e01c18ce1e14696077f3c8e72d67d', 'function': 'original_001', 'default': True, 'branch': 'reproduce'}, {'path': '2DCNN_test/ensure/PCA_2DCNN_test_fraze.py', 'sha256': '296d5f7e43667951f7669a578e92d997b9e91bd95650de3b67d2666e57d6f328', 'function': 'original_003', 'default': True, 'branch': 'reproduce'}, {'path': 'RF_test/PCA_CNN_RF_test.py', 'sha256': '87231b20011a4eab7c295e9b4be20f53a719a3cb8f8f9469f52c71a7cb8295f9', 'function': 'original_079', 'default': True, 'branch': 'reproduce'}, {'path': 'Revised/Original_RF_ROC.py', 'sha256': '9f7a73981904ea3018a5b03476a45da8cf7f49e26cf556a8ef08fe4eb14a9da4', 'function': 'original_074', 'default': True, 'branch': 'reproduce'}, {'path': 'Revised/PCA_RF_ROC.py', 'sha256': '05fa0acc26419d5edb6934fa161844fd3b822079dbdaccf8dd238a8e26b47a20', 'function': 'original_077', 'default': True, 'branch': 'reproduce'}, {'path': 'Revised/Original_1DCNN_ROC.py', 'sha256': '915bc0eaa8b3aa5e7a5c7f20b27420e203ad661b61756070e0c798d17d8c3e81', 'function': 'original_072', 'default': True, 'branch': 'reproduce'}, {'path': 'Revised/PCA_1DCNN_ROC.py', 'sha256': '40c993657b6a4e87d97240ccc9bdd0fb2ade78841c61e481d8460496450d6804', 'function': 'original_075', 'default': True, 'branch': 'reproduce'}, {'path': 'Revised/Original_2DCNN_ROC.py', 'sha256': '69f4616fd7c3e0fb962a410f0922fb1e507e323e1ab8c05f1af14a880189d245', 'function': 'original_073', 'default': True, 'branch': 'reproduce'}, {'path': 'Revised/PCA_2DCNN_ROC.py', 'sha256': '69e14890e5a5058aa4911c43562973990cc7495ca5a901f615b154780e81f4d3', 'function': 'original_076', 'default': True, 'branch': 'reproduce'}, {'path': 'PCA_test/PCA_test_2025_03_01/PCA-TEST_03_01.py', 'sha256': '036770e04398485875dd98fe098f5d390da9dd085a4caeda4778342fbd4c8b72', 'function': 'original_069', 'default': True, 'branch': 'preprocess'}, {'path': 'PCA_test/PCA_Mapping.py', 'sha256': '418c8a8c65877895708f06aee650ebcde208d757d112a2b2d372c5fe26db1033', 'function': 'original_068', 'default': True, 'branch': 'preprocess'}, {'path': 'CNN_Grad_Excel_PCA_mapping_label.py', 'sha256': 'ad9bf741a24760c93b463ebd8f37067f0b00d6e7d93ba57b00688f56f767f762', 'function': 'original_037', 'default': False, 'branch': 'preprocess'}, {'path': '1DCNN_test/PCA_CNN_1DCNN_test.py', 'sha256': '0d3e10732fe3300a756209a8a31c7bf8de927a81d3fdaffceb64dda01c1e5656', 'function': 'original_000', 'default': False, 'branch': 'training'}, {'path': 'original_analyses/1DCNN/original_1DCNN_test.py', 'sha256': '78d9e416745b8646916a00e51e65abe25532dd524150e5117ead1ba332a6b519', 'function': 'original_054', 'default': False, 'branch': 'training'}, {'path': 'original_analyses/original_2DCNN_test.py', 'sha256': '546065c891d319a618255695dd649744479c3648bb8e4b7ed0764ec6ae1bd581', 'function': 'original_055', 'default': False, 'branch': 'training'}, {'path': 'PCA_CNN_2DCNN.py', 'sha256': '04b31e6a868afce4410c507a84a153a18d108fb39453fb4d512a777a1977d6bb', 'function': 'original_060', 'default': False, 'branch': 'training'}, {'path': 'PCA_test/Mafic rock_only/mafic_only_PCA_test.py', 'sha256': 'bdfa77f5ecbc04eca03e3fbacd02505be45ca764e40150b63c2789867a5c1627', 'function': 'original_066', 'default': True, 'branch': 'supplementary'}, {'path': 'PCA_test/Zircon_only_PCA/Zircon_only_PCA_test.py', 'sha256': 'c419def9efc3fea4a0fc061066de22ebed02aabce3d789eede0a7fc5b49a9c9a', 'function': 'original_070', 'default': True, 'branch': 'supplementary'}, {'path': 'PCA_test/Zircon_only_PCA/监督-PLS-DA/Zircon_only_pls-da_test.py', 'sha256': 'd3ec85cec440289633d3fae9fcec8ad184b4bc89c365dec3f798b1f0d9df31d1', 'function': 'original_071', 'default': False, 'branch': 'supplementary'}, {'path': 'U-Pb_Best_age/U_Pb_Best_age.py', 'sha256': '5f02c0029bf6d47bf126ba86c96236badac03d0c1fd60c3b6ee49c26ab4352b8', 'function': 'original_083', 'default': False, 'branch': 'supplementary'}, {'path': 'U-Pb_Best_age/U_Pb_Best_age_and_period.py', 'sha256': '509dd95dbe9495cdac96d73856ceb9e45f2b3026e5a5a6613cac7b7cb5b02419', 'function': 'original_084', 'default': False, 'branch': 'supplementary'}, {'path': 'U-Pb_Best_age/U_Pb_Best_age_figure.py', 'sha256': '668ee8cc524d6fafa35ff32ba6ce04102d9ec807114cd80e072a6873b51cac34', 'function': 'original_085', 'default': False, 'branch': 'supplementary'}, {'path': 'Data_analyses/Time_list.py', 'sha256': '42ec8ce947fc4ff4379c959aa8463b278c8159ca67c40a6e985eb8fb65a26f98', 'function': 'original_052', 'default': False, 'branch': 'supplementary'}, {'path': 'Data_analyses/Locaiton analyses_01_17.py', 'sha256': '354de4bfa6c3621c51810bf047e88ccb318766eff650942e9888fcc205c34723', 'function': 'original_050', 'default': False, 'branch': 'supplementary'}, {'path': 'Data_analyses/Date_Th_U/Th_U_Best_age.py', 'sha256': 'df2c96a2e59bd02ecb34909e0522ae5bbc0c4169f7a49be0a824efd0f757e9db', 'function': 'original_042', 'default': False, 'branch': 'supplementary'}, {'path': 'Data_analyses/Date_Th_U/Th_U_Best_age_combine_mafic_date.py', 'sha256': 'bda1c95f3d8abf5ab719ef77b05302cdae62c8986cf907b70b3c66382cf9d548', 'function': 'original_043', 'default': False, 'branch': 'supplementary'}, {'path': 'Data_analyses/Date_Th_U/Th_U_Best_age_only_orogenic_gold_ and_disrtubution.py', 'sha256': '0ea6a0d6bebf3ff979bc99a47aae2b640d201b9322c66f65d6ae5f9afc8769c0', 'function': 'original_047', 'default': False, 'branch': 'supplementary'}, {'path': 'Th_U_and_temperature/Magamatic_Th_U_and_Temperature.py', 'sha256': 'ea8e8ffc2c0564d064e557a75a880259abd736fbfc49629371c1ea3238288c2a', 'function': 'original_082', 'default': False, 'branch': 'supplementary'}, {'path': '相关性分析/Normality test.py', 'sha256': '03f2609e055ac127eff42c319d022477028660045e7b74172b1c343fdc221ec4', 'function': 'original_089', 'default': False, 'branch': 'supplementary'}]
PREPARED_PCA = OUT / 'PREP__PCA_recomputed.csv'

def prepare_complete():
    """Reuse saved PCA/labels exactly. No PCA fit and no random sampling."""
    import pandas as pd
    import numpy as np
    import hashlib, shutil
    OUT.mkdir(parents=True, exist_ok=True)

    def read(path):
        data = pd.read_csv(path)
        data.columns = data.columns.str.strip()
        return data.rename(columns={'X': 'x', 'Y': 'y'})
    pcs = [f'PCA{i}' for i in range(1, 24)]
    train_path = DATA_ROOT / 'data/PCA_with_labels_Train.csv'
    test_path = DATA_ROOT / 'data/PCA_with_labels_Test.csv'
    train = read(train_path)
    test = read(test_path)
    for data in [train, test]:
        if not set(['x', 'y', 'Label'] + pcs).issubset(data.columns):
            raise ValueError('Missing saved PCA columns')
        if data.duplicated(['x', 'y']).any():
            raise ValueError('Duplicate coordinates')
    joined = train.merge(test, on=['x', 'y'], how='left', suffixes=('_train', '_test'), validate='one_to_one', indicator=True)
    same_coordinates = bool((joined['_merge'] == 'both').all())
    same_labels = bool(np.array_equal(joined.Label_train.to_numpy(), joined.Label_test.to_numpy(), equal_nan=True))
    same_pca = bool(np.array_equal(joined[[c + '_train' for c in pcs]].to_numpy(), joined[[c + '_test' for c in pcs]].to_numpy(), equal_nan=True))
    if not all([same_coordinates, same_labels, same_pca]):
        raise ValueError('Saved PCA train/test mismatch; see source data')
    labelled = test.loc[test.Label.isin([0, 1]), ['x', 'y', 'Label']].copy()
    labelled.to_csv(OUT / 'SAVED__all_labelled_pixel_coordinates.csv', index=False, encoding='utf-8-sig')
    labelled.loc[labelled.Label == 0].to_csv(OUT / 'SAVED__negative_pixel_coordinates.csv', index=False, encoding='utf-8-sig')
    labelled.loc[labelled.Label == 1].to_csv(OUT / 'SAVED__positive_pixel_coordinates.csv', index=False, encoding='utf-8-sig')
    records = []
    for src, name in [(train_path, 'SAVED__PCA_with_labels_Train.csv'), (test_path, 'SAVED__PCA_with_labels_Test.csv')]:
        dst = OUT / name
        shutil.copyfile(src, dst)

        def digest(path):
            h = hashlib.sha256()
            with open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(4194304), b''):
                    h.update(block)
            return h.hexdigest()
        source_hash = digest(src)
        copy_hash = digest(dst)
        if source_hash != copy_hash:
            raise IOError('Copy hash mismatch')
        records.append(dict(source=str(src), output=str(dst), sha256=source_hash, identical_bytes=True))
    report = dict(same_coordinates=same_coordinates, same_labels=same_labels, same_PCA_values=same_pca, train_rows=len(train), full_grid_rows=len(test), positive_pixels=int((labelled.Label == 1).sum()), negative_pixels=int((labelled.Label == 0).sum()), files=records)
    (OUT / 'SAVED__input_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    (OUT / 'SAVED__methods.txt').write_text('Default: use historical PCA and label CSV exactly. Negative pixel coordinates are recovered from Label==0, not sampled again. Original sampling/PCA code remains in optional modules for method inspection. Recomputed PCA or labels must not replace frozen model inputs. Window-center coordinates are not inferred from expanded pixel labels. Copied PCA files are verified byte-for-byte by SHA256.', encoding='utf8')
    print('Saved PCA and label inputs verified:', report['train_rows'], 'training rows;', report['negative_pixels'], 'negative pixels')
    saved_pca_results()
    restore_archived_epoch_figures()

def install_dual_image_output():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    import weakref
    if getattr(Figure.savefig, '_single_pair', False):
        return
    chosen = configure_chinese_fonts()
    original = Figure.savefig
    saved = weakref.WeakKeyDictionary()

    def dual(self, fname, *args, **kwargs):
        if not isinstance(fname, (str, bytes, os.PathLike)):
            return original(self, fname, *args, **kwargs)
        target = Path(_output(fname))
        if target.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.svg']:
            return original(self, target, *args, **kwargs)
        key = str(target.with_suffix(''))
        done = saved.setdefault(self, set())
        if key in done:
            return
        for text in self.findobj(match=Text):
            families = text.get_fontfamily()
            if chosen not in families:
                text.set_fontfamily(families + [chosen])
        for ext in ['jpg', 'svg']:
            options = dict(kwargs)
            options['format'] = ext
            original(self, target.with_suffix('.' + ext), *args, **options)
        done.add(key)
    dual._single_pair = True
    Figure.savefig = dual

def saved_pca_results():
    global REFIT_SCORES
    REFIT_SCORES, refit_components, refit_variance = refit_pca()
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    global ACTIVE_STAGE
    ACTIVE_STAGE = 'PCA_SAVED'
    original_068()
    source = DATA_ROOT / 'PCA/PCA_components.xlsx'
    components = refit_components.copy()
    components.to_csv(OUT / 'PCA_REFIT__all_coefficients.csv', encoding='utf-8-sig')
    report = []
    for i in range(1, 24):
        column = f'PCA{i}'
        values = pd.to_numeric(components[column], errors='raise').to_numpy()
        order = np.argsort(values)
        fig, ax = plt.subplots(figsize=(6, 10))
        ax.barh(components.index.astype(str).to_numpy()[order], values[order], color=['blue' if v > 0 else 'red' for v in values[order]], edgecolor='black')
        ax.axvline(0, color='black', linewidth=1, linestyle='-')
        ax.set_ylabel('Features', fontsize=24, fontweight='bold')
        ax.set_xlabel('Contribution score', fontsize=24, fontweight='bold')
        ax.set_title(f'PCA{i} feature contributions', fontsize=24, fontweight='bold')
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(OUT / f'PCA_REFIT__PCA{i}_coefficients.svg')
        plt.close(fig)
    variance_path = DATA_ROOT / 'PCA_test/PCA_test/PCA_variance_contributions.xlsx'
    if variance_path.exists():
        variance = refit_variance.copy()
        variance.to_csv(OUT / 'PCA_REFIT__variance_table.csv', index=False, encoding='utf-8-sig')
        fig, ax = plt.subplots()
        ax.bar(range(1, len(variance) + 1), variance['Eigenvalues'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Eigenvalue')
        other = ax.twinx()
        other.plot(range(1, len(variance) + 1), variance['Cumulative Variance (%)'], 'o-', color='red')
        other.set_ylabel('Cumulative variance (%)')
        other.set_ylim(0, 110)
        fig.savefig(OUT / 'PCA_REFIT__scree.svg')
        plt.close(fig)
        report.append(dict(role='archived scree table', source=str(variance_path), note='Archived table version retained; not asserted to be fitted by frozen-model preprocessing'))
    report.append(dict(role='coefficient table', source=str(source)))
    report.append(dict(role='PCA map scores', source=str(DATA_ROOT / 'PCA/PCA.csv')))
    (OUT / 'PCA_REFIT__sources.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')

def restore_archived_epoch_figures():
    """Historical graphs, explicitly labelled; not recovered training histories."""
    import shutil
    from PIL import Image
    sources = {'Original_1DCNN_accuracy': 'original_analyses/1DCNN/accuracy_curve_origin_1DCNN', 'Original_1DCNN_loss': 'original_analyses/1DCNN/loss_curve_origin_1DCNN', 'PCA_1DCNN_accuracy': '1DCNN_test/ensure/accuracy_curve_1DCNN_test', 'PCA_1DCNN_loss': '1DCNN_test/ensure/loss_curve_1DCNN_test', 'Original_2DCNN_accuracy': 'original_analyses/accuracy_curve_original_analyse', 'Original_2DCNN_loss': 'original_analyses/loss_curve_original_analyse', 'PCA_2DCNN_accuracy': '2DCNN_test/ensure/accuracy_curve_2DCNN_test', 'PCA_2DCNN_loss': '2DCNN_test/ensure/loss_curve_2DCNN_test'}
    report = []
    for name, relative in sources.items():
        base = DATA_ROOT / relative
        svg = base.with_suffix('.svg')
        png = base.with_suffix('.png')
        if not svg.exists() or not png.exists():
            raise FileNotFoundError(str(base))
        shutil.copyfile(svg, OUT / f'ARCHIVED_EPOCH__{name}.svg')
        with Image.open(png) as im:
            im.convert('RGB').save(OUT / f'ARCHIVED_EPOCH__{name}.jpg', quality=95)
        report.append(dict(name=name, svg_source=str(svg), raster_source=str(png), status='archived original graph; not a newly trained curve', jpg_note='JPEG converted from original PNG; lossy raster conversion'))
    (OUT / 'ARCHIVED_EPOCH__provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
INPUT_MANIFEST = [{'relative': '1DCNN_test/ensure/accuracy_curve_1DCNN_test.png', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3533']}, {'relative': '1DCNN_test/ensure/accuracy_curve_1DCNN_test.svg', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3532']}, {'relative': '1DCNN_test/ensure/loss_curve_1DCNN_test.png', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3533']}, {'relative': '1DCNN_test/ensure/loss_curve_1DCNN_test.svg', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3532']}, {'relative': '1DCNN_test/ensure/trained_model_1DCNN_test.h5', 'used_by': ['original_001:186', 'original_075:2301']}, {'relative': '2DCNN_test/ensure/2DCNN_model.h5', 'used_by': ['original_003:289', 'original_076:2596']}, {'relative': '2DCNN_test/ensure/accuracy_curve_2DCNN_test.png', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3533']}, {'relative': '2DCNN_test/ensure/accuracy_curve_2DCNN_test.svg', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3532']}, {'relative': '2DCNN_test/ensure/loss_curve_2DCNN_test.png', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3533']}, {'relative': '2DCNN_test/ensure/loss_curve_2DCNN_test.svg', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3532']}, {'relative': 'PCA/PCA.csv', 'used_by': ['original_068:1250']}, {'relative': 'PCA/PCA_components.xlsx', 'used_by': ['saved_pca_results:3484']}, {'relative': 'PCA_test/PCA_test/PCA_variance_contributions.xlsx', 'used_by': ['saved_pca_results:3503']}, {'relative': 'RF_test/predicted_values_RF_test.csv', 'used_by': ['original_079:3048']}, {'relative': 'data/PCA_with_labels_Test.csv', 'used_by': ['original_001:189', 'original_003:238', 'original_075:2245', 'original_075:2289', 'original_076:2490', 'original_077:2764', 'original_077:2814', 'original_079:3038', 'prepare_complete:3435']}, {'relative': 'data/PCA_with_labels_Train.csv', 'used_by': ['original_077:2763', 'original_077:2812', 'original_079:3037', 'prepare_complete:3435']}, {'relative': 'data/Sample Location_delet_orogenic_gold.xlsx', 'used_by': ['original_080:3090']}, {'relative': 'original_analyses/1DCNN/accuracy_curve_origin_1DCNN.png', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3533']}, {'relative': 'original_analyses/1DCNN/accuracy_curve_origin_1DCNN.svg', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3532']}, {'relative': 'original_analyses/1DCNN/loss_curve_origin_1DCNN.png', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3533']}, {'relative': 'original_analyses/1DCNN/loss_curve_origin_1DCNN.svg', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3532']}, {'relative': 'original_analyses/1DCNN/trained_model_origin_1DCNN.h5', 'used_by': ['original_072:1599']}, {'relative': 'original_analyses/2DCNN_model_original_analyse.h5', 'used_by': ['original_056:914', 'original_073:1869']}, {'relative': 'original_analyses/accuracy_curve_original_analyse.png', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3533']}, {'relative': 'original_analyses/accuracy_curve_original_analyse.svg', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3532']}, {'relative': 'original_analyses/grayscale_data_combined_test.csv', 'used_by': ['original_056:918', 'original_072:1559', 'original_073:1771', 'original_074:2022']}, {'relative': 'original_analyses/grayscale_data_combined_train.csv', 'used_by': ['original_056:917', 'original_072:1558', 'original_073:1770', 'original_074:2021']}, {'relative': 'original_analyses/loss_curve_original_analyse.png', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3533']}, {'relative': 'original_analyses/loss_curve_original_analyse.svg', 'used_by': ['restore_archived_epoch_figures', 'restore_archived_epoch_figures:3532']}, {'relative': 'PCA/grayscale_data_combined.csv', 'used_by': ['refit_pca']}]

def configure_chinese_fonts():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    choices = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'SimSun']
    chosen = next((n for n in choices if n in names), None)
    if chosen is None:
        raise RuntimeError('未找到中文字体，请安装微软雅黑/黑体/Noto Sans CJK SC')
    plt.rcParams['font.family'] = ['DejaVu Sans', chosen]
    plt.rcParams['axes.unicode_minus'] = False
    return chosen

def compare_coordinate_values(new, old, columns, name, allow_sign=False):
    import numpy as np
    import pandas as pd
    for data in [new, old]:
        data.rename(columns={'X': 'x', 'Y': 'y'}, inplace=True)
        if data.duplicated(['x', 'y']).any():
            raise ValueError('重复坐标不能直接比较')
    merged = new[['x', 'y'] + columns].merge(old[['x', 'y'] + columns], on=['x', 'y'], suffixes=('_new', '_old'), how='outer', indicator=True, validate='one_to_one')
    common = merged['_merge'] == 'both'
    rows = []
    for c in columns:
        a = merged.loc[common, c + '_new'].to_numpy()
        b = merged.loc[common, c + '_old'].to_numpy()
        finite = np.isfinite(a) & np.isfinite(b)
        x = a[finite]
        y = b[finite]
        sign = -1 if allow_sign and len(x) and (np.sum(x * y) < 0) else 1
        rows.append(dict(feature=c, common_coordinates=int(common.sum()), new_only=int((merged['_merge'] == 'left_only').sum()), old_only=int((merged['_merge'] == 'right_only').sum()), equal_values=bool(np.array_equal(a, b, equal_nan=True)), same_nan_pattern=bool(np.array_equal(np.isnan(a), np.isnan(b))), max_abs_difference=float(np.max(np.abs(x - y))) if len(x) else None, RMSE=float(np.sqrt(np.mean((x - y) ** 2))) if len(x) else None, diagnostic_sign=sign, sign_aligned_max_abs_difference=float(np.max(np.abs(sign * x - y))) if len(x) else None))
    pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')
    return rows

def refit_pca():
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    source = DATA_ROOT / 'PCA/grayscale_data_combined.csv'
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip()
    features = df.loc[:, '176Hf-177Hf':'Hf(t)']
    z = (features - features.mean()) / features.std()
    if features.shape[1] != 23 or not np.isfinite(z.to_numpy()).all():
        raise ValueError('PCA输入无效')
    model = PCA(n_components=23, svd_solver='full')
    values = model.fit_transform(z)
    reference = pd.read_csv(DATA_ROOT / 'PCA/PCA.csv')
    reference.columns = reference.columns.str.strip()
    reference = reference.rename(columns={'x': 'X', 'y': 'Y'})
    if reference.duplicated(['X', 'Y']).any():
        raise ValueError('Reference PCA contains duplicate coordinates')
    index = pd.MultiIndex.from_frame(reference[['X', 'Y']])
    positions = index.get_indexer(pd.MultiIndex.from_frame(df[['X', 'Y']]))
    if (positions < 0).any():
        raise ValueError('PCA reference does not cover all input coordinates')
    signs = []
    diagnostics = []
    for j in range(23):
        old_values = reference[f'PCA{j + 1}'].to_numpy()[positions]
        sign = -1 if np.dot(values[:, j], old_values) < 0 else 1
        values[:, j] *= sign
        model.components_[j, :] *= sign
        signs.append(sign)
        diagnostics.append(dict(PC=j + 1, applied_sign=sign, max_abs_difference=float(np.max(np.abs(values[:, j] - old_values)))))
    pd.DataFrame(diagnostics).to_csv(OUT / 'PCA_REFIT__sign_alignment.csv', index=False)
    pcs = [f'PCA{i}' for i in range(1, 24)]
    new = df[['X', 'Y']].copy()
    new[pcs] = values
    new.to_csv(OUT / 'PCA_REFIT__scores.csv', index=False)
    components = pd.DataFrame(model.components_.T, index=features.columns, columns=pcs)
    components.to_csv(OUT / 'PCA_REFIT__coefficients.csv', encoding='utf-8-sig')
    variance = pd.DataFrame({'Principal Component': pcs, 'Eigenvalues': model.explained_variance_, 'Explained Variance Ratio': model.explained_variance_ratio_, 'Cumulative Variance (%)': np.cumsum(model.explained_variance_ratio_) * 100})
    variance.to_csv(OUT / 'PCA_REFIT__variance.csv', index=False)
    import joblib
    joblib.dump(dict(pca=model, mean=features.mean(), std=features.std(), features=list(features)), OUT / 'PCA_REFIT__transform.joblib')
    old = pd.read_csv(DATA_ROOT / 'PCA/PCA.csv')
    compare_coordinate_values(new.copy(), old, pcs, 'PCA_REFIT__vs_saved_scores', True)
    model_input = pd.read_csv(DATA_ROOT / 'data/PCA_with_labels_Test.csv')
    compare_coordinate_values(new.copy(), model_input, pcs, 'PCA_REFIT__vs_frozen_model_inputs', True)
    return (new, components, variance)

def install_input_audit():
    """Reject source file reads outside DATA_ROOT; log every supported input API."""
    import pandas as pd
    import tensorflow as tf
    import shutil
    from PIL import Image
    events = []

    def check(value, api):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return value
        path = Path(os.fsdecode(value))
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        try:
            path.relative_to(DATA_ROOT.resolve())
        except ValueError:
            raise ValueError(f'输入来源不在data目录：{api}: {path}')
        events.append(dict(api=api, path=str(path)))
        (OUT / 'input_read_audit.json').write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding='utf8')
        return value
    for name in ['read_csv', 'read_excel']:
        original = getattr(pd, name)

        def wrapper(value, *args, _original=original, _name=name, **kwargs):
            return _original(check(value, _name), *args, **kwargs)
        setattr(pd, name, wrapper)
    load = tf.keras.models.load_model

    def load_model(path, *args, **kwargs):
        return load(check(path, 'load_model'), *args, **kwargs)
    tf.keras.models.load_model = load_model
    image_open = Image.open

    def open_image(path, *args, **kwargs):
        return image_open(check(path, 'Image.open'), *args, **kwargs)
    Image.open = open_image
    copy = shutil.copyfile

    def copyfile(source, destination, *args, **kwargs):
        return copy(check(source, 'copyfile.source'), destination, *args, **kwargs)
    shutil.copyfile = copyfile

def export_prediction_map(data, probabilities, name):
    import numpy as np
    import matplotlib.pyplot as plt
    result = data[['x', 'y']].copy()
    result['Predicted_Probability'] = np.asarray(probabilities).ravel()
    result.to_csv(OUT / (name + '__full_predictions.csv'), index=False)
    finite = np.isfinite(result.Predicted_Probability.to_numpy())
    if finite.sum() < 3:
        raise ValueError('预测地图有效点不足')
    fig, ax = plt.subplots(figsize=(18, 5.64), dpi=100)
    contour = ax.tricontourf(result.loc[finite, 'x'], result.loc[finite, 'y'], result.loc[finite, 'Predicted_Probability'], levels=14, cmap='coolwarm')
    fig.colorbar(contour, ax=ax, label='Predicted Probability')
    ax.axis('off')
    ax.set_title('Predicted Probability Contour Map ' + name)
    fig.savefig(OUT / (name + '__contour_map.svg'), dpi=100, bbox_inches='tight')
    plt.close(fig)

def export_pca_sensitivity(values, name):
    import pandas as pd
    import matplotlib.pyplot as plt
    pd.DataFrame({'PCA': [f'PCA{i + 1}' for i in range(23)], 'mean_absolute_probability_change': values}).to_csv(OUT / (name + '__sensitivity.csv'), index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(23), values, color='skyblue', edgecolor='black')
    ax.set_xlabel('PCA Feature Index')
    ax.set_ylabel('Average Contribution to Prediction')
    ax.set_title('PCA Feature Sensitivity Analysis - ' + name)
    fig.savefig(OUT / (name + '__sensitivity.svg'), bbox_inches='tight')
    plt.close(fig)
PA_RESULTS = {}

def export_pa_summary(data, name, scope):
    """Export P-A results calculated by this run, not cached manuscript plots."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    required = ['Prospectivity_score_class', 'Predicted_area_percent', 'Captured_positive_samples_percent']
    if not set(required).issubset(data.columns):
        raise ValueError('Missing P-A columns')
    d = data.copy()
    d['Model'] = name
    d['Statistical_scope'] = scope
    d.to_csv(OUT / f'PA__{name}__data.csv', index=False, encoding='utf-8-sig')
    PA_RESULTS[name] = d
    audit = []
    for model, frame in PA_RESULTS.items():
        area = float(frame.Predicted_area_percent.sum())
        capture = float(frame.Captured_positive_samples_percent.sum())
        audit.append(dict(model=model, area_sum_percent=area, capture_sum_percent=capture, area_sum_is_100=bool(np.isclose(area, 100)), capture_sum_is_100=bool(np.isclose(capture, 100)), statistical_scope=frame.Statistical_scope.iloc[0]))
    pd.DataFrame(audit).to_csv(OUT / 'PA__scope_and_checks.csv', index=False, encoding='utf-8-sig')

    def draw(ax, frame, title):
        x = np.arange(len(frame))
        width = 0.38
        ax.bar(x - width / 2, frame.Predicted_area_percent, width, label='Predicted area (%)', edgecolor='black', color='tab:blue')
        ax.bar(x + width / 2, frame.Captured_positive_samples_percent, width, label='Captured positive samples (%)', edgecolor='black', color='tab:orange')
        ax.set_xticks(x)
        ax.set_xticklabels(frame.Prospectivity_score_class, rotation=45, ha='right')
        ax.set_xlabel('Prospectivity score class')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(title)
        ax.grid(True, axis='y')
        ax.legend(fontsize=8)
    fig, ax = plt.subplots(figsize=(10, 6))
    draw(ax, d, name + ' Prediction-Area Plot')
    fig.tight_layout()
    fig.savefig(OUT / f'PA__{name}__plot.svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    order = ['Original_RF', 'PCA_RF', 'Original_1DCNN', 'PCA_1DCNN', 'Original_2DCNN', 'PCA_2DCNN']
    if all((model in PA_RESULTS for model in order)):
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        for ax, model in zip(axes.ravel(), order):
            draw(ax, PA_RESULTS[model], model)
        fig.tight_layout()
        fig.savefig(OUT / 'PA__six_models__comparison.svg', dpi=300, bbox_inches='tight')
        plt.close(fig)
        pd.concat([PA_RESULTS[model] for model in order], ignore_index=True).to_csv(OUT / 'PA__six_models__data.csv', index=False, encoding='utf-8-sig')
    print('P-A exported:', name, '; scope:', scope, '; file:', OUT / f'PA__{name}__plot.jpg')
ROC_RESULTS = {}

def export_roc_auc(name, fpr, tpr, thresholds, roc_auc, y_true, scope):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    thresholds = np.asarray(thresholds)
    if len(fpr) != len(tpr) or len(fpr) != len(thresholds):
        raise ValueError('ROC arrays have inconsistent lengths')
    if not np.isfinite(roc_auc) or not 0 <= roc_auc <= 1:
        raise ValueError('Invalid AUC')
    curve = pd.DataFrame({'False_positive_rate': fpr, 'True_positive_rate': tpr, 'Threshold': thresholds})
    curve.to_csv(OUT / f'ROC__{name}__curve_data.csv', index=False, encoding='utf-8-sig')
    row = dict(Model=name, ROC_AUC=float(roc_auc), Samples=len(y_true), Positive_samples=int(np.sum(np.asarray(y_true) == 1)), Negative_samples=int(np.sum(np.asarray(y_true) == 0)), Evaluation_unit=scope, Independence='Historical labelled data; independent holdout not established')
    ROC_RESULTS[name] = (curve, row)
    pd.DataFrame([row]).to_csv(OUT / f'ROC__{name}__AUC.csv', index=False, encoding='utf-8-sig')
    summary = pd.DataFrame([item[1] for item in ROC_RESULTS.values()])
    summary.to_csv(OUT / 'ROC__AUC_summary.csv', index=False, encoding='utf-8-sig')
    (OUT / 'ROC__AUC_summary.txt').write_text(summary.to_string(index=False), encoding='utf8')

    def draw(ax, curve, row):
        ax.plot(curve.False_positive_rate, curve.True_positive_rate, linewidth=2, label=f'AUC = {row['ROC_AUC']:.4f}')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Random prediction')
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(row['Model'])
        ax.grid(True)
        ax.legend(fontsize=9)
    fig, ax = plt.subplots(figsize=(7, 6))
    draw(ax, curve, row)
    fig.tight_layout()
    fig.savefig(OUT / f'ROC__{name}__curve.svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    order = ['Original_RF', 'PCA_RF', 'Original_1DCNN', 'PCA_1DCNN', 'Original_2DCNN', 'PCA_2DCNN']
    if all((model in ROC_RESULTS for model in order)):
        fig, axes = plt.subplots(3, 2, figsize=(13, 16))
        for ax, model in zip(axes.ravel(), order):
            draw(ax, *ROC_RESULTS[model])
        fig.tight_layout()
        fig.savefig(OUT / 'ROC__six_models__panels.svg', dpi=300, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 8))
        for model in order:
            data, metrics = ROC_RESULTS[model]
            ax.plot(data.False_positive_rate, data.True_positive_rate, label=f'{model}: {metrics['ROC_AUC']:.4f}')
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        fig.savefig(OUT / 'ROC__six_models__overlay.svg', dpi=300, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        values = [ROC_RESULTS[model][1]['ROC_AUC'] for model in order]
        ax.bar(order, values)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('ROC-AUC')
        ax.tick_params(axis='x', rotation=25)
        for i, value in enumerate(values):
            ax.text(i, value + 0.01, f'{value:.4f}', ha='center')
        fig.tight_layout()
        fig.savefig(OUT / 'ROC__six_models__AUC_comparison.svg', dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f'ROC/AUC已输出：{name}, AUC={roc_auc:.6f}, 统计单位={scope}')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='筛选后的原工程单文件。默认复用原CSV及冻结模型；不调用外部Python脚本。')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--branch', choices=['reproduce', 'preprocess', 'training', 'supplementary', 'all'], default='all')
    parser.add_argument('--stage', help='清单内原始相对路径；只执行一个模块')
    args = parser.parse_args()
    if args.list:
        for item in CATALOG:
            print(item['branch'] + ' | ' + item['path'])
        return
    if args.stage or args.branch in ['training', 'supplementary']:
        parser.error('此精确依赖包覆盖默认all/reproduce/preprocess流程。训练与补充分支不在本次复制范围，不能直接运行。')
    missing = [str(DATA_ROOT / entry['relative']) for entry in INPUT_MANIFEST if not (DATA_ROOT / entry['relative']).is_file()]
    if missing:
        raise FileNotFoundError('请先运行01复制脚本，缺少输入：\n' + '\n'.join(missing))
    if args.stage:
        tasks = [i for i in CATALOG if i['path'] == args.stage]
        if not tasks:
            parser.error('未保留此任务，请用--list查看')
    else:
        tasks = [i for i in CATALOG if i['branch'] == args.branch]
    OUT.mkdir(parents=True, exist_ok=True)
    install_dual_image_output()
    install_input_audit()
    (OUT / 'selected_source_catalog.json').write_text(json.dumps(CATALOG, ensure_ascii=False, indent=2), encoding='utf8')
    if not args.stage and args.branch in ['all', 'preprocess']:
        prepare_complete()
    if not args.stage and args.branch == 'all':
        tasks = [i for i in CATALOG if i['branch'] == 'reproduce']
    if not args.stage and args.branch == 'preprocess':
        tasks = []
    report = []
    for item in tasks:
        print('运行原模块:', item['path'])
        try:
            run_stage(item)
            report.append(dict(source=item['path'], status='completed'))
        except Exception:
            report.append(dict(source=item['path'], status='failed', error=traceback.format_exc()))
            raise
        finally:
            (OUT / 'selected_execution_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('所选任务执行结束；结果位于', OUT)

# 后续审稿分析。保留上方原始流程，各新增阶段使用独立解释器。
REVIEW_MODULES = {'PA_saved_six_models': 'import os\n"""Reanalyse saved predictions only. No model loading, fitting or PCA fitting."""\nimport argparse,json,subprocess,hashlib\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport matplotlib\nmatplotlib.use(\'Agg\')\nimport matplotlib.pyplot as plt\nplt.rcParams.update({\'font.family\':\'Times New Roman\',\'svg.fonttype\':\'none\'})\nARCHIVE=\'C:/central asian mapping.rar\'\nFILES={\n\'Original_RF\':\'Revised/fig/original-RF_predictions.csv\',\n\'PCA_RF\':\'Revised/fig/PCA-RF_predictions.csv\',\n\'Original_1DCNN\':\'original_analyses/1DCNN/PCA_with_predictions_origin_1DCNN.csv\',\n\'PCA_1DCNN\':\'Revised/fig/PCA-1DCNN_predictions.csv\',\n\'Original_2DCNN\':\'Revised/fig/original-2DCNN_predictions.csv\',\n\'PCA_2DCNN\':\'2DCNN_test/ensure/PCA_with_full_predictions_Test_2DCNN_test.csv\'}\nROOT=Path(os.environ[\'CAOB_PROJECT\'])/\'data\'\n\ndef save(fig,out,name):\n    for ext in [\'jpg\',\'svg\']:fig.savefig(out/(name+\'.\'+ext),dpi=300,bbox_inches=\'tight\')\n    plt.close(fig)\n\ndef analyse(frame,points,out,name):\n    frame=frame.rename(columns={\'X\':\'x\',\'Y\':\'y\'})\n    candidates=[c for c in frame if \'probab\' in c.lower() or c==\'Predicted_Probability\']\n    if len(candidates)!=1:raise ValueError(f\'{name}: ambiguous probability columns {candidates}\')\n    col=candidates[0]\n    frame[col]=pd.to_numeric(frame[col],errors=\'coerce\')\n    if frame.duplicated([\'x\',\'y\']).any():raise ValueError(\'Duplicate pixel coordinates\')\n    valid=np.isfinite(frame[col]);missing=int((~valid).sum());data=frame.loc[valid,[\'x\',\'y\',col]].copy()\n    scores=data[col].to_numpy(); ss=np.sort(scores)\n    if len(ss)==0:raise ValueError(\'No finite predictions\')\n    if ss.min()<0 or ss.max()>1:raise ValueError(\'Scores outside probability interval\')\n    lookup=data.set_index([\'x\',\'y\'])[col]\n    deposits=points.copy();deposits[\'x\']=((deposits.Longitude-55)/80*1799).astype(int);deposits[\'y\']=((deposits.Latitude-35)/25*563).astype(int)\n    deposits[\'probability\']=[lookup.get((x,y),np.nan) for x,y in zip(deposits.x,deposits.y)]\n    deposits.to_csv(out/\'deposit_scores.csv\',index=False)\n    ds=np.sort(deposits.probability.dropna().to_numpy())\n    if not len(ds):raise ValueError(\'No deposits have predictions\')\n    # All observed score cutoffs, with ties kept together; no arbitrary 0.1 probability bins.\n    thresholds=np.unique(np.r_[0.,scores,np.nextafter(scores.max(),np.inf)])\n    area=(len(ss)-np.searchsorted(ss,thresholds,side=\'left\'))/len(ss)*100\n    capture=(len(ds)-np.searchsorted(ds,thresholds,side=\'left\'))/len(ds)*100\n    density=np.divide(capture,area,out=np.full(len(area),np.nan),where=area>0)\n    curve=pd.DataFrame({\'threshold\':thresholds,\'occupied_area_percent\':area,\'deposit_capture_percent\':capture,\'normalized_density\':density})\n    curve.to_csv(out/\'PA_all_thresholds.csv\',index=False)\n    # Reversed right axis: visual intersection satisfies Pr = 100 - Oa.\n    delta=capture+area-100\n    idx=np.flatnonzero(delta>=0)[-1];j=min(idx+1,len(delta)-1)\n    weight=0 if delta[idx]==delta[j] else delta[idx]/(delta[idx]-delta[j])\n    cross_t=thresholds[idx]+weight*(thresholds[j]-thresholds[idx])\n    pr=capture[idx]+weight*(capture[j]-capture[idx]);oa=area[idx]+weight*(area[j]-area[idx])\n    actual_a=float((scores>=cross_t).mean()*100);actual_p=float((ds>=cross_t).mean()*100)\n    row=dict(model=name,intersection_threshold_interpolated=cross_t,prediction_rate_interpolated=pr,occupied_area_interpolated=oa,normalized_density_interpolated=pr/oa if oa else None,actual_threshold_area_percent=actual_a,actual_threshold_deposit_capture_percent=actual_p,actual_normalized_density=actual_p/actual_a if actual_a else None,total_map_rows=len(frame),finite_map_pixels=len(scores),missing_prediction_pixels=missing,total_deposits=len(deposits),deposits_with_prediction=len(ds),deposits_missing_prediction=len(deposits)-len(ds))\n    fig,ax=plt.subplots(figsize=(10,6));right=ax.twinx()\n    ax.plot(thresholds,capture,color=\'tab:blue\',label=\'Deposit capture\');right.plot(thresholds,area,color=\'tab:orange\',label=\'Occupied area\')\n    ax.set_ylim(0,100);right.set_ylim(100,0);ax.set_xlim(0,1)\n    ax.set_xlabel(\'Prospectivity threshold (score >= threshold)\');ax.set_ylabel(\'Prediction / success rate (%)\');right.set_ylabel(\'Occupied area (%) — reversed\')\n    ax.axvline(cross_t,color=\'gray\',ls=\'--\');ax.set_title(name+\' P–A (known-deposit success assessment)\')\n    ax.legend(loc=\'upper left\');right.legend(loc=\'upper right\');save(fig,out,\'PA\')\n    fig,ax=plt.subplots(figsize=(7,6));ax.plot(area[::-1],capture[::-1],color=\'tab:blue\');ax.plot([0,100],[0,100],\'--\',color=\'gray\');ax.set(xlabel=\'Occupied area (%)\',ylabel=\'Known deposits captured (%)\',title=name+\' success-rate curve\',xlim=(0,100),ylim=(0,100));save(fig,out,\'success_rate\')\n    (out/\'metrics.txt\').write_text(\'\\n\'.join(f\'{k} = {v}\' for k,v in row.items())+\'\\nDescriptive assessment using known deposits, not independent validation.\\nArea fractions assume equal-area pixels. Missing predictions are excluded, not filled.\\nThe interpolated crossing may not be exactly attainable with discrete/tied scores.\\n\',encoding=\'utf-8-sig\')\n    return row\n\ndef main():\n    p=argparse.ArgumentParser();p.add_argument(\'--output\',default=\'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/fig/P-C\');p.add_argument(\'--archive\',default=ARCHIVE);a=p.parse_args()\n    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)\n    points=pd.read_excel(ROOT/\'data/Sample Location_delet_orogenic_gold.xlsx\')\n    rows=[];manifest=[]\n    for name,member in FILES.items():\n        out=root/name;out.mkdir(exist_ok=True);cached=out/\'saved_predictions.csv\'\n        cached=Path(a.archive)/member\n        manifest.append(dict(model=name,archive_member=member,sha256=hashlib.sha256(cached.read_bytes()).hexdigest()))\n        rows.append(analyse(pd.read_csv(cached),points,out,name));print(name,\'complete\',flush=True)\n    pd.DataFrame(rows).to_csv(root/\'six_models_summary.csv\',index=False)\n    (root/\'input_manifest.json\').write_text(json.dumps(manifest,indent=2),encoding=\'utf8\')\n    (root/\'method_notes.txt\').write_text(\'Reference: Yousefi & Carranza, Computers & Geosciences 79 (2015), 69–81, sections 2.3–2.4.\\nAdaptation: cumulative score-threshold sweep for continuous model outputs, rather than reproducing the paper\\\'s fractal discretization of evidence layers. Prediction rate counts deposit records, not expanded positive pixels. Nd=Pr/Oa. Opposing Y axes intersect where Pr+Oa=100.\\nCrossing thresholds are descriptive map delineation, not independently optimized or validated classification thresholds. Report full curves and missing-data coverage; do not compare conditional coverage as if identical. Pixel area assumed equal; georeferenced area weights are needed for true geographical area. No three/five-class probability scheme is imposed.\\n\',encoding=\'utf8\')\nif __name__==\'__main__\':main()\n\n', 'PAC_manuscript': '"""Audit and supplement the archived manuscript evaluation. No model training.\nRun with Python and numpy/pandas/matplotlib. Outputs remain descriptive.\n"""\nfrom pathlib import Path\nimport sys, subprocess, io, argparse\nsys.path.insert(0, str(Path(__file__).parent/\'python_dependencies\'))\nimport numpy as np\nif not hasattr(np, \'trapezoid\'): np.trapezoid=np.trapz\nimport pandas as pd\nimport matplotlib\nmatplotlib.use(\'Agg\')\nimport matplotlib.pyplot as plt\nplt.rcParams.update({\'font.family\':\'Times New Roman\',\'svg.fonttype\':\'none\'})\nNAMES=[\'original-RF\',\'PCA-RF\',\'original-1DCNN\',\'PCA-1DCNN\',\'original-2DCNN\',\'PCA-transformed 2DCNN\']\n\ndef main(archive, output):\n output.mkdir(parents=True,exist_ok=True)\n def read(member):\n  return (archive/member).read_bytes()\n rows=[]; fig,axes=plt.subplots(2,3,figsize=(17,10))\n for name in NAMES:\n  patch=\'2DCNN\' in name\n  member=(\'2DCNN_test/ensure/PCA_with_full_predictions_Test_2DCNN_test.csv\' if name==NAMES[-1] else \'Revised/fig/\'+name+\'_predictions.csv\')\n  # Read only coordinates, labels and probabilities, avoiding unnecessary feature columns.\n  cache=archive/member\n  d=pd.read_csv(cache,usecols=lambda c:c.lower() in [\'x\',\'y\',\'label\'] or \'probab\' in c.lower()).rename(columns={\'X\':\'x\',\'Y\':\'y\'})\n  col=[c for c in d if \'probab\' in c.lower()]\n  if len(col)!=1: raise ValueError((name,col))\n  d=d.rename(columns={col[0]:\'score\'})\n  if patch:\n   xmin,ymin=d.x.min(),d.y.min(); xmax,ymax=d.x.max(),d.y.max()\n   d[\'bx\']=((d.x-xmin)//9).astype(int);d[\'by\']=((d.y-ymin)//9).astype(int)\n   d=d[(xmin+d.bx*9+8<=xmax)&(ymin+d.by*9+8<=ymax)]\n   g=d.groupby([\'bx\',\'by\'])\n   e=g.agg(n=(\'Label\',\'size\'),Label=(\'Label\',lambda v:int((v==1).any())),score=(\'score\',\'first\'),low=(\'score\',\'min\'),high=(\'score\',\'max\'))\n   e=e[e.n>=50]\n   if ((e.high-e.low)>1e-7).any(): raise ValueError(\'Scores vary within original patch grid: \'+name)\n   e=e[np.isfinite(e.score)]\n   unit=\'9x9 patch (any positive pixel)\'\n  else:\n   e=d[d.Label.isin([0,1])&np.isfinite(d.score)].copy();unit=\'labelled pixel\'\n  y=e.Label.to_numpy(int);s=e.score.to_numpy(float)\n  e.to_csv(output/(name+\'_evaluation_samples.csv\'),index=False)\n  # Complete tied-score groups; no arbitrary ordering within equal scores.\n  order=np.argsort(-s,kind=\'stable\');sy=s[order];yy=y[order]\n  ends=np.r_[np.flatnonzero(np.diff(sy)!=0),len(sy)-1]\n  tp=np.r_[0,np.cumsum(yy)[ends]];fp=np.r_[0,(ends+1)-np.cumsum(yy)[ends]]\n  thresholds=np.r_[np.inf,sy[ends]];rec=tp/y.sum();fpr=fp/(len(y)-y.sum())\n  precision=np.divide(tp,tp+fp,out=np.ones(len(tp)),where=tp+fp>0)\n  auc=float(np.trapezoid(rec,fpr));ap=float(np.sum(np.diff(rec)*precision[1:]))\n  saved=pd.read_csv(io.BytesIO(read(\'Revised/fig/\'+name+\'_ROC_curve_data.csv\')))\n  reference=float(np.trapezoid(saved.iloc[:,1],saved.iloc[:,0]))\n  match=abs(auc-reference)<5e-6\n  ba=.5*((y[s>=.5]==1).sum()/y.sum()+(y[s<.5]==0).sum()/(len(y)-y.sum()))\n  rows.append(dict(model=name,evaluation_unit=unit,n=len(y),positives=int(y.sum()),ROC_AUC=auc,archived_ROC_AUC=reference,ROC_reproduced=match,BA_at_0_5=ba,AP=ap,PR_AUC=float(np.trapezoid(precision,rec)),source=member))\n  label=name+(\' [MISMATCH]\' if not match else \'\')\n  axes[0,0].plot(fpr,rec,label=label)\n  axes[0,1].plot(rec,precision,label=label)\n  ts=np.linspace(0,1,201);bs=[.5*(np.mean(s[y==1]>=t)+np.mean(s[y==0]<t)) for t in ts]\n  axes[0,2].plot(ts,bs,label=label)\n  # Success/PA use the same evaluation units as ROC. These are sample fractions,\n  # not geographic occupied areas when evaluation covers only labelled pixels.\n  fraction=(tp+fp)/len(y);lift=np.divide(rec,fraction,out=np.full(len(rec),np.nan),where=fraction>0)\n  axes[1,0].plot(fraction*100,rec*100,label=label)\n  axes[1,1].plot(fraction*100,lift,label=label)\n  axes[1,2].plot(thresholds[1:],rec[1:]*100,label=label)\n  pd.DataFrame(dict(threshold=thresholds,recall=rec,FPR=fpr,precision=precision,evaluation_unit_fraction=fraction,lift=lift)).to_csv(output/(name+\'_curves.csv\'),index=False)\n  print(name,\'AUC\',auc,\'reference\',reference,\'match\',match,flush=True)\n  # Map-area P-A: labelled positive pixels for pixel models; positive patches\n  # for patch models, matching each original script\'s positive-unit definition.\n  population=s if patch else d.loc[np.isfinite(d.score),\'score\'].to_numpy()\n  cut=np.unique(np.r_[0,population,np.nextafter(population.max(),np.inf)])\n  area=100*(len(population)-np.searchsorted(np.sort(population),cut))/len(population)\n  positives=np.sort(s[y==1]);capture=100*(len(positives)-np.searchsorted(positives,cut))/len(positives)\n  density=np.divide(capture,area,out=np.full(len(area),np.nan),where=area>0)\n  delta=capture+area-100;k=np.flatnonzero((delta[:-1]>=0)&(delta[1:]<=0))[-1]\n  w=delta[k]/(delta[k]-delta[k+1]) if delta[k]!=delta[k+1] else 0\n  pr=capture[k]+w*(capture[k+1]-capture[k]);oa=area[k]+w*(area[k+1]-area[k])\n  pf,pa=plt.subplots(figsize=(10,6));right=pa.twinx()\n  pa.plot(cut,capture,color=\'tab:blue\',label=\'Positive-unit capture\');right.plot(cut,area,color=\'tab:orange\',label=\'Occupied area\')\n  pa.set(xlim=(0,1),ylim=(0,100),xlabel=\'Prospectivity threshold\',ylabel=\'Positive-unit capture (%)\',title=name+\' P–A\');right.set(ylim=(100,0),ylabel=\'Occupied area (%) — reversed\')\n  pa.legend(loc=\'upper left\');right.legend(loc=\'upper right\')\n  pa.text(.04,.08,f\'Interpolated intersection\\nPrediction rate = {pr:.2f}%\\nOccupied area = {oa:.3f}%\\nNormalized density = {pr/oa:.2f}\',transform=pa.transAxes,bbox=dict(facecolor=\'white\',edgecolor=\'0.7\'))\n  for ext in [\'png\',\'svg\',\'jpg\']:pf.savefig(output/(name+\'_PA.\'+ext),dpi=220,bbox_inches=\'tight\')\n  plt.close(pf)\n  pd.DataFrame(dict(threshold=cut,occupied_area_percent=area,positive_unit_capture_percent=capture,normalized_density=density)).to_csv(output/(name+\'_PA_data.csv\'),index=False)\n pd.DataFrame(rows).to_csv(output/\'metrics_and_ROC_audit.csv\',index=False)\n for ax,title,x,ylabel in zip(axes.flat,[\'ROC\',\'Precision–recall\',\'Balanced accuracy\',\'Success (ROC evaluation units)\',\'Lift (ROC evaluation units)\',\'Cumulative capture\'],[\'False positive rate\',\'Recall\',\'Threshold\',\'Selected evaluation units (%)\',\'Selected evaluation units (%)\',\'Threshold\'],[\'True positive rate\',\'Precision\',\'Balanced accuracy\',\'Positive units captured (%)\',\'Lift\',\'Positive units captured (%)\']):\n  ax.set(title=title,xlabel=x,ylabel=ylabel);ax.grid(alpha=.2)\n axes[1,1].set_yscale(\'log\');axes[0,0].plot([0,1],[0,1],\'--\',color=\'gray\')\n fig.legend(*axes[0,0].get_legend_handles_labels(),loc=\'lower center\',ncol=3)\n fig.tight_layout(rect=(0,.09,1,1))\n for ext in [\'png\',\'svg\',\'jpg\']:fig.savefig(output/(\'PAC_manuscript_audit.\'+ext),dpi=220,bbox_inches=\'tight\')\n (output/\'README.txt\').write_text(\'ROC/PR/BA use archived manuscript evaluation units. 2DCNN uses nonoverlapping 9x9 patches, any positive pixel => positive patch, min 50 pixels. Other models use binary-labelled pixels. No common-coverage filtering. Success/lift here use these SAME units; X is selected evaluation-unit percentage, NOT geographical area. Geographic P-A must be evaluated separately. Models have different units and populations; this reproduces legacy evaluation, not a unified independent validation. MISMATCH marks predictions that do not reproduce archived ROC; do not use those as manuscript-matched results. PCA 2DCNN saved-map source must pass this audit before use.\\n\',encoding=\'utf8\')\n\nif __name__==\'__main__\':\n p=argparse.ArgumentParser();p.add_argument(\'--archive\',type=Path,default=Path(\'C:/central asian mapping.rar\'));p.add_argument(\'--output\',type=Path,default=Path(__file__).parent/\'PAC_manuscript_results\');a=p.parse_args();main(a.archive,a.output)\n', 'PAC_publication_four': '"""Four publication figures from manuscript-matched evaluation samples.\n\nFirst run PAC_manuscript.py to rebuild and audit the evaluation samples.\nThis script performs no training and never changes scores or labels.\n"""\nfrom pathlib import Path\nimport os, sys, argparse\nBASE=Path(__file__).parent\nos.environ.setdefault(\'MPLCONFIGDIR\',str(BASE/\'mpl_config\'))\nsys.path.insert(0,str(BASE/\'python_dependencies\'))\nimport numpy as np\nif not hasattr(np, \'trapezoid\'): np.trapezoid=np.trapz\nimport pandas as pd\nimport matplotlib\nmatplotlib.use(\'Agg\')\nimport matplotlib.pyplot as plt\nfrom matplotlib.lines import Line2D\nfrom matplotlib.ticker import MultipleLocator\n\nMODELS=[\'original-RF\',\'PCA-RF\',\'original-1DCNN\',\'PCA-1DCNN\',\'original-2DCNN\',\'PCA-transformed 2DCNN\']\nLABELS=[\'Original RF\',\'PCA RF\',\'Original 1-D CNN\',\'PCA 1-D CNN\',\'Original 2-D CNN\',\'PCA 2-D CNN\']\n# Model family is encoded by colour; PCA is distinguished by a dashed line.\nCOLORS=[\'#2166AC\',\'#2166AC\',\'#238B45\',\'#238B45\',\'#B35806\',\'#B35806\']\nplt.rcParams.update({\'font.family\':\'Times New Roman\',\'font.size\':12,\n \'axes.titlesize\':17,\'axes.labelsize\':14,\'legend.fontsize\':11,\n \'axes.linewidth\':.9,\'svg.fonttype\':\'none\',\'pdf.fonttype\':42,\n \'xtick.direction\':\'out\',\'ytick.direction\':\'out\',\'savefig.facecolor\':\'white\'})\n\ndef style_axes(ax):\n # Keep all four borders visible, with quiet gridlines behind the data.\n for spine in ax.spines.values():\n  spine.set_visible(True)\n  spine.set_color(\'#404852\')\n  spine.set_linewidth(.9)\n ax.grid(color=\'#E8EBEF\', linewidth=.6)\n ax.set_axisbelow(True)\n ax.tick_params(direction=\'out\', length=4, width=.8, colors=\'#303740\', pad=5)\n ax.xaxis.labelpad=9\n ax.yaxis.labelpad=9\n\n\ndef main(source,output):\n output.mkdir(parents=True,exist_ok=True)\n audit=pd.read_csv(source/\'metrics_and_ROC_audit.csv\').set_index(\'model\')\n if not audit.ROC_reproduced.all():raise ValueError(\'ROC audit failed; resolve input mismatch first\')\n figures={}\n for kind in [\'ROC\',\'Balanced_accuracy\',\'Success_rate\',\'PR_AUC\']:\n  fig,ax=plt.subplots(figsize=(8.2,7.3));fig.subplots_adjust(left=.13,right=.96,top=.90,bottom=.30)\n  style_axes(ax)\n  ax.grid(color=\'#E3E7EB\',linewidth=.65);ax.set_axisbelow(True)\n  figures[kind]=(fig,ax)\n rows=[]\n for i,(model,label,color) in enumerate(zip(MODELS,LABELS,COLORS)):\n  d=pd.read_csv(source/(model+\'_evaluation_samples.csv\'));y=d.Label.to_numpy(int);s=d.score.to_numpy(float)\n  c=pd.read_csv(source/(model+\'_curves.csv\'))\n  roc=float(np.trapezoid(c.recall,c.FPR))\n  if abs(roc-audit.loc[model,\'archived_ROC_AUC\'])>5e-6:raise ValueError(model+\' ROC mismatch\')\n  pra=float(np.trapezoid(c.precision,c.recall))\n  style=dict(color=color,lw=2.0,ls=\'--\' if i%2 else \'-\',alpha=.95)\n  figures[\'ROC\'][1].plot(c.FPR,c.recall,label=f\'{label}   AUC = {roc:.4f}\',**style)\n  figures[\'PR_AUC\'][1].plot(c.recall,c.precision,label=f\'{label}   PR-AUC = {pra:.4f}\',**style)\n  ts=np.linspace(0,1,501)\n  ba=np.array([.5*(np.mean(s[y==1]>=t)+np.mean(s[y==0]<t)) for t in ts])\n  ba05=.5*(np.mean(s[y==1]>=.5)+np.mean(s[y==0]<.5))\n  figures[\'Balanced_accuracy\'][1].plot(ts,ba,label=f\'{label}   BA at 0.5 = {ba05:.4f}\',**style)\n  figures[\'Success_rate\'][1].plot(c.evaluation_unit_fraction*100,c.recall*100,label=label,**style)\n  rows.append(dict(model=model,ROC_AUC=roc,PR_AUC_trapezoidal=pra,AP=audit.loc[model,\'AP\'],balanced_accuracy_at_0_5=ba05,n=len(y),positive_prevalence=y.mean()))\n  pd.DataFrame(dict(threshold=ts,balanced_accuracy=ba)).to_csv(output/(model+\'_BA.csv\'),index=False)\n configs={\n \'ROC\':(\'ROC curves\',\'False positive rate\',\'True positive rate\'),\n \'Balanced_accuracy\':(\'Balanced accuracy\',\'Classification threshold\',\'Balanced accuracy\'),\n \'Success_rate\':(\'Success-rate curves\',\'Selected evaluation units (%)\',\'Positive units captured (%)\'),\n \'PR_AUC\':(\'Precision–recall curves\',\'Recall\',\'Precision\')}\n # Symmetric four-panel layout; lift omitted because it duplicates success data.\n combined=plt.figure(figsize=(13.2,10.6))\n grid=combined.add_gridspec(2,2,left=.085,right=.975,top=.935,bottom=.16,wspace=.26,hspace=.32)\n slots=[grid[0,0],grid[0,1],grid[1,0],grid[1,1]]\n combined_order=[\'ROC\',\'Balanced_accuracy\',\'Success_rate\',\'PR_AUC\']\n for kind,(fig,ax) in figures.items():\n  title,xlab,ylab=configs[kind];ax.set(title=title,xlabel=xlab,ylabel=ylab)\n  ax.set_xlim(0,100 if kind==\'Success_rate\' else 1)\n  ax.set_ylim(0,102 if kind==\'Success_rate\' else 1.025)\n  ax.xaxis.set_major_locator(MultipleLocator(20 if kind==\'Success_rate\' else .2))\n  ax.yaxis.set_major_locator(MultipleLocator(20 if kind==\'Success_rate\' else .2))\n  if kind==\'ROC\':ax.plot([0,1],[0,1],color=\'#87929C\',lw=1,ls=\':\',zorder=0)\n  if kind==\'Success_rate\':ax.plot([0,100],[0,100],color=\'#87929C\',lw=1,ls=\':\',zorder=0)\n  if kind==\'Balanced_accuracy\':\n   ax.axvline(.5,color=\'#87929C\',lw=1,ls=\':\',zorder=0)\n   ax.axhline(.5,color=\'#B7BEC5\',lw=.8,ls=\':\',zorder=0)\n  ax.legend(loc=\'upper center\',bbox_to_anchor=(.5,-.19),ncol=2,frameon=False,handlelength=3,columnspacing=1.5,labelspacing=.8)\n  for ext in [\'png\',\'jpg\',\'svg\',\'pdf\']:fig.savefig(output/(kind+\'.\'+ext),dpi=350)\n  # Copy plotted data and styles, never rasterize the single-panel images.\n  position=combined_order.index(kind)\n  target=combined.add_subplot(slots[position])\n  for line in ax.lines:\n   target.plot(line.get_xdata(),line.get_ydata(),color=line.get_color(),\n               linestyle=line.get_linestyle(),linewidth=1.9 if line.get_linewidth()>1 else .85,\n               alpha=line.get_alpha(),label=line.get_label())\n  target.set(xscale=ax.get_xscale(),yscale=ax.get_yscale(),xlim=ax.get_xlim(),ylim=ax.get_ylim(),\n             xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel())\n  target.set_title(title,fontsize=15,pad=13)\n  target.xaxis.label.set_size(11);target.yaxis.label.set_size(11)\n  target.tick_params(labelsize=10)\n  style_axes(target)\n  target.grid(color=\'#E8EBEF\',linewidth=.6);target.set_axisbelow(True)\n  target.text(-.105,1.035,f\'({chr(97+position)})\',transform=target.transAxes,\n              fontsize=15,fontweight=\'bold\',va=\'bottom\')\n  if kind in [\'ROC\',\'PR_AUC\',\'Balanced_accuracy\']:\n   field={\'ROC\':\'ROC_AUC\',\'PR_AUC\':\'PR_AUC_trapezoidal\',\'Balanced_accuracy\':\'balanced_accuracy_at_0_5\'}[kind]\n   heading={\'ROC\':\'AUC\',\'PR_AUC\':\'PR-AUC\',\'Balanced_accuracy\':\'BA at 0.5\'}[kind]\n   annotations=\'\\n\'.join(f\'{label}: {row[field]:.4f}\' for label,row in zip(LABELS,rows))\n   target.text(.97,.06 if kind!=\'PR_AUC\' else .32,heading+\'\\n\'+annotations,\n               transform=target.transAxes,ha=\'right\',va=\'bottom\',fontsize=9,\n               linespacing=1.4,bbox=dict(boxstyle=\'round,pad=0.6,rounding_size=0.15\',facecolor=\'white\',alpha=.96,edgecolor=\'#D2D8DF\',linewidth=.8))\n  plt.close(fig)\n handles=[Line2D([0],[0],color=color,lw=2,ls=\'--\' if i%2 else \'-\',label=label)\n          for i,(label,color) in enumerate(zip(LABELS,COLORS))]\n combined.legend(handles=handles,loc=\'lower center\',bbox_to_anchor=(.5,.025),\n                 ncol=3,frameon=False,fontsize=12,handlelength=3,columnspacing=2.5,labelspacing=.9)\n for ext in [\'png\',\'jpg\',\'svg\',\'pdf\']:combined.savefig(output/(\'All_four_metrics.\'+ext),dpi=300)\n plt.close(combined)\n pd.DataFrame(rows).to_csv(output/\'metrics.csv\',index=False)\n (output/\'README.txt\').write_text(\'Four panels: ROC, balanced accuracy, success rate, precision-recall. Lift omitted because it is derived from success rate and extreme small-sample values reduce readability. ROC-AUC verified against archived manuscript ROC. Original/PCA share family colours with solid/dashed lines. PR-AUC is trapezoidal; AP separately reported. BA threshold=0.5. Success uses ROC evaluation units: labelled pixels for RF/1DCNN, 9x9 patches for 2DCNN. X is sample fraction, not geographic area. Different populations and prevalences limit comparison, especially PR-AUC. Legacy descriptive evaluation, not independent validation. No smoothing or changes to scores.\\n\',encoding=\'utf8\')\n print(pd.DataFrame(rows).to_string(index=False))\n\nif __name__==\'__main__\':\n p=argparse.ArgumentParser();p.add_argument(\'--input\',type=Path,default=BASE/\'PAC_manuscript_results\');p.add_argument(\'--output\',type=Path,default=BASE/\'PAC_four_figures\');a=p.parse_args();main(a.input,a.output)\n', 'refresh_repeat_outputs': '"""Refresh saved repeated-run outputs without fitting or loading models."""\nfrom pathlib import Path\nimport argparse, importlib.util, shutil\nimport numpy as np\nimport pandas as pd\nimport matplotlib\nmatplotlib.use(\'Agg\')\nimport matplotlib.pyplot as plt\nfrom sklearn.metrics import roc_curve, average_precision_score\n\nSRC=Path(__file__).resolve().parent\nDEFAULT=SRC.parent/\'fig/Repeat\'\ndef load(name):\n    spec=importlib.util.spec_from_file_location(name,SRC/(name+\'.py\'))\n    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m\n\ndef refresh(run,output=None):\n    pa=load(\'PA_saved_six_models\');pub=load(\'PAC_publication_four\')\n    # NumPy 1.x compatibility for the supplied publication script.\n    if not hasattr(np,\'trapezoid\'): np.trapezoid=np.trapz\n    out=Path(output) if output else run/\'latest_figures\'\n    out.mkdir(parents=True,exist_ok=True)\n    ev=out/\'evaluation\';ev.mkdir(exist_ok=True)\n    points=pd.read_excel(pa.ROOT/\'data/Sample Location_delet_orogenic_gold.xlsx\')\n    summaries=[];audits=[]\n    keys=[\'Original_RF\',\'PCA_RF\',\'Original_1DCNN\',\'PCA_1DCNN\',\'Original_2DCNN\',\'PCA_2DCNN\']\n    archived=pd.read_csv(run/\'ROC__AUC_summary.csv\').set_index(\'Model\')\n    for i,(model,key) in enumerate(zip(pub.MODELS,keys)):\n        path=run/\'results/Revised/fig\'/(model+\'_predictions.csv\')\n        if i==5: path=run/\'results/data/PCA_with_full_predictions_Test_2DCNN.csv\'\n        cols=pd.read_csv(path,nrows=0).columns\n        scorecol=next(c for c in cols if \'probab\' in c.lower())\n        d=pd.read_csv(path,usecols=[\'x\',\'y\',\'Label\',scorecol]).rename(columns={scorecol:\'score\'})\n        target=out/key;target.mkdir(exist_ok=True)\n        mapdata=d\n        if i==2:\n            # The ROC export contains labelled pixels only; use the full training-stage map for P-A.\n            full=run/\'results/original_analyses/1DCNN/PCA_with_predictions_origin_1DCNN.csv\'\n            mapdata=pd.read_csv(full,usecols=[\'x\',\'y\',\'Label\',\'Predicted_Probability\']).rename(columns={\'Predicted_Probability\':\'score\'})\n        summaries.append(pa.analyse(mapdata.rename(columns={\'score\':\'Predicted_Probability\'}),points,target,key))\n        if i<2:\n            grid=np.full((564,1800),np.nan)\n            grid[d.y.astype(int),d.x.astype(int)]=d.score\n            fig,ax=plt.subplots(figsize=(15,5))\n            im=ax.imshow(grid,origin=\'lower\',cmap=\'coolwarm\',vmin=0,vmax=1,extent=(55,135,35,60),aspect=\'auto\',rasterized=True)\n            ax.set(xlabel=\'Longitude (°E)\',ylabel=\'Latitude (°N)\',title=pub.LABELS[i])\n            fig.colorbar(im,ax=ax,label=\'Predicted probability\')\n            for ext in [\'jpg\',\'svg\']:fig.savefig(target/(\'prediction_map.\'+ext),dpi=200,bbox_inches=\'tight\')\n            plt.close(fig)\n        if i>=4:\n            # Match legacy ROC: complete 9x9 tiles; any positive pixel => positive tile.\n            z=d[(d.x<1800)&(d.y<558)].copy()\n            z[\'bx\']=(z.x//9).astype(int);z[\'by\']=(z.y//9).astype(int)\n            g=z.groupby([\'by\',\'bx\'],sort=True)\n            e=g.agg(Label=(\'Label\',lambda a:int(a.eq(1).any())),score=(\'score\',\'first\'),n=(\'score\',\'count\'),lo=(\'score\',\'min\'),hi=(\'score\',\'max\'))\n            expected=64 if i==5 else 81\n            if not (e.n.eq(expected)&(e.hi-e.lo).abs().lt(1e-7)).all():raise ValueError(key+\' nonconstant/incomplete tile scores\')\n            e=e[[\'Label\',\'score\']]\n        else:e=d.loc[d.Label.isin([0,1]),[\'Label\',\'score\']]\n        if not np.isfinite(e.score).all():raise ValueError(key+\' missing evaluation scores\')\n        y=e.Label.to_numpy(int);s=e.score.to_numpy(float)\n        f,t,th=roc_curve(y,s,drop_intermediate=False)\n        auc=float(np.trapezoid(t,f));old=float(archived.loc[key,\'ROC_AUC\'])\n        if abs(auc-old)>5e-6:raise ValueError(f\'{key}: reconstructed AUC {auc} != saved {old}\')\n        order=np.argsort(-s,kind=\'stable\');ys=y[order];ss=s[order]\n        ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]\n        tp=np.cumsum(ys)[ends];fp=ends+1-tp\n        recall=np.r_[0,tp/y.sum()];precision=np.r_[1,tp/(tp+fp)]\n        # ROC/PR/success arrays use all distinct thresholds and keep ties together.\n        pd.DataFrame(dict(FPR=np.r_[0,fp/(len(y)-y.sum())],recall=recall,precision=precision,evaluation_unit_fraction=np.r_[0,(ends+1)/len(y)])).to_csv(ev/(model+\'_curves.csv\'),index=False)\n        e.to_csv(ev/(model+\'_evaluation_samples.csv\'),index=False)\n        audits.append(dict(model=model,ROC_reproduced=True,archived_ROC_AUC=old,AP=average_precision_score(y,s)))\n    pd.DataFrame(audits).to_csv(ev/\'metrics_and_ROC_audit.csv\',index=False)\n    pd.DataFrame(summaries).to_csv(out/\'PA_summary.csv\',index=False)\n    pub.main(ev,out/\'publication_four\')\n    from organize_repeat_outputs import organize\n    organize(run,out)\n    print(\'REFRESHED\',run.name,flush=True)\n\nif __name__==\'__main__\':\n    p=argparse.ArgumentParser();p.add_argument(\'--base\',type=Path,default=DEFAULT);p.add_argument(\'--runs\',nargs=\'*\');p.add_argument(\'--output\',type=Path)\n    a=p.parse_args()\n    for run in sorted(a.base.glob(\'run_*\')):\n        if run.is_dir() and (not a.runs or run.name in a.runs):refresh(run,a.output/run.name if a.output else None)\n', 'organize_repeat_outputs': '"""Create explicitly named display outputs; retain legacy files used by training/evaluation."""\nfrom pathlib import Path\nimport shutil,json\n\nMODELS={\'Original_RF\':\'Original_RF\',\'PCA_RF\':\'PCA_RF\',\'Original_1DCNN\':\'Original_1D_CNN\',\'PCA_1DCNN\':\'PCA_1D_CNN\',\'Original_2DCNN\':\'Original_2D_CNN\',\'PCA_2DCNN\':\'PCA_2D_CNN\'}\nMAPS={\n\'Original_1DCNN\':\'original_analyses/1DCNN/contour_map_origin_1DCNN\',\n\'PCA_1DCNN\':\'1DCNN_test/contour_map_1DCNN_test\',\n\'Original_2DCNN\':\'original_analyses/contour_map_original_analyse\',\n\'PCA_2DCNN\':\'contour_map_2DCNN\'}\n\ndef organize(run,latest=None):\n    run=Path(run);latest=Path(latest) if latest else run/\'latest_figures\'\n    root=latest/\'Model_Figures\';root.mkdir(exist_ok=True)\n    manifest=[]\n    def copy(src,dst):\n        if src.is_file():\n            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)\n            manifest.append({\'source\':str(src),\'output\':str(dst)})\n    for old,name in MODELS.items():\n        dest=root/name;dest.mkdir(exist_ok=True)\n        for ext in [\'jpg\',\'svg\']:\n            src=(latest/old/(\'prediction_map.\'+ext)) if old not in MAPS else (run/\'results\'/(MAPS[old]+\'.\'+ext))\n            copy(src,dest/(name+\'__Prediction_Map.\'+ext))\n        folder=latest/old\n        if folder.exists():\n            for src in folder.iterdir():\n                if src.is_file() and not src.name.startswith(\'prediction_map.\'):\n                    copy(src,dest/(name+\'__\'+src.name))\n        for kind in [\'ROC\',\'PA\']:\n            for src in run.glob(kind+\'__\'+old+\'__*\'):\n                copy(src,dest/(name+\'__Legacy_\'+kind+\'__\'+src.name.split(\'__\',2)[-1]))\n    pub=latest/\'publication_four\'\n    if pub.exists():\n        for src in pub.iterdir():\n            if src.is_file():copy(src,root/\'Six_Models_Comparison\'/(\'Six_Models__\'+src.name))\n    for name in [\'PA_summary.csv\']:\n        copy(latest/name,root/\'Six_Models_Comparison\'/(\'Six_Models__\'+name))\n    (root/\'file_manifest.json\').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding=\'utf-8\')\n    (root/\'README.txt\').write_text(\'Original = original geochemical features; PCA = PCA-transformed features.\\nPrediction_Map = spatial prediction map. PA = new threshold P-A plot. Legacy = old evaluation plots retained for traceability.\\nSix_Models_Comparison contains the latest combined figures. Source files remain in their original locations for script compatibility.\\n\',encoding=\'utf-8\')\n    return len(manifest)\n\nif __name__==\'__main__\':\n    import argparse\n    p=argparse.ArgumentParser();p.add_argument(\'--base\',type=Path,required=True);a=p.parse_args()\n    for run in sorted(a.base.glob(\'run_*\')):\n        if run.is_dir():print(run.name,organize(run),flush=True)\n', 'fixed_labels_spatial_cv': '"""Fixed original labels only; spatial block + buffer CV, no negative resampling.\nModels retain original fixed hyperparameters; true spatial 9x9 inputs replace legacy row packing.\n"""\nimport os\nos.environ[\'TF_CPP_MIN_LOG_LEVEL\']=\'2\'\nfrom pathlib import Path\nimport json,gc\nimport numpy as np,pandas as pd\nfrom scipy.ndimage import binary_erosion,label as components\nfrom sklearn.model_selection import StratifiedGroupKFold\nfrom sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,precision_recall_curve,auc\nfrom spatial_two_schemes import ROOT,RAW,geo,distance,fit,predict\nOUT=Path(os.environ[\'CAOB_SPATIAL\']);OUT.mkdir(parents=True,exist_ok=True)\ndef main():\n grids={};masks={}\n for branch,rel,cols in [(\'Raw\',\'original_analyses/grayscale_data_combined_test.csv\',RAW),(\'PCA\',\'data/PCA_with_labels_Test.csv\',[f\'PCA{i}\' for i in range(1,24)])]:\n  d=pd.read_csv(ROOT/\'data\'/rel).sort_values([\'y\',\'x\']);grids[branch]=d[cols].to_numpy(np.float32).reshape(564,1800,23);masks[branch]=d.Label.to_numpy().reshape(564,1800)\n assert np.array_equal(masks[\'Raw\'],masks[\'PCA\'],equal_nan=True),\'Raw/PCA labels differ: stop rather than alter labels\'\n mask=masks[\'Raw\'];sites=[];ys=[]\n # One full 9x9 window per connected same-label region; all 81 labels must agree.\n # Deterministic central representative avoids treating neighboring cells as independent deposits.\n for cls in [1,0]:\n  lab,n=components(mask==cls,structure=np.ones((3,3)))\n  valid=binary_erosion(mask==cls,structure=np.ones((9,9)))\n  for k in range(1,n+1):\n   yy,xx=np.where((lab==k)&valid)\n   if not len(xx):continue\n   cy,cx=np.mean(np.where(lab==k),axis=1);i=np.argmin((xx-cx)**2+(yy-cy)**2);sites.append([xx[i],yy[i]]);ys.append(cls)\n xy=np.array(sites);y=np.array(ys);cells=np.floor(geo(xy)/5).astype(int);groups=np.unique(cells,axis=0,return_inverse=True)[1]\n splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(xy,y,groups));folds=np.full(len(y),-1);plans=[]\n # Exclude training windows with ANY constituent cell <=100 km from ANY test cell.\n offsets=np.array([(dx,dy) for dy in range(-4,5) for dx in range(-4,5)])\n for k,(tr,te) in enumerate(splits):\n  testcells=(xy[te,None,:]+offsets).reshape(-1,2);keep=[];mins=[]\n  for i in tr:\n   ds=distance(xy[i]+offsets,testcells).min()\n   if ds>100:keep.append(i);mins.append(float(ds))\n  tr=np.array(keep,int);assert len(np.unique(y[tr]))==2 and len(np.unique(y[te]))==2\n  assert not set(groups[tr])&set(groups[te]);folds[te]=k;plans.append((tr,te))\n  pd.DataFrame(xy[tr],columns=[\'x\',\'y\']).assign(label=y[tr],minimum_test_cell_km=mins).to_csv(OUT/f\'fold_{k+1}_train_centers.csv\',index=False)\n pd.DataFrame(xy,columns=[\'x\',\'y\']).assign(label=y,fold=folds).to_csv(OUT/\'fixed_sites.csv\',index=False)\n (OUT/\'protocol.json\').write_text(json.dumps(dict(negative_labels=\'unchanged original CSV labels; no new sampling or PUB\',unit=\'one complete same-label 9x9 window per connected labeled region\',block_degrees=5,folds=5,buffer_km=100,buffer_definition=\'minimum over all training and test window cells\',PCA=\'existing frozen full-grid PCA, transductive; not fold-refitted\',hyperparameters=\'fixed historical original model values; no test-based tuning\',two_dimensional=\'original architecture; true spatial windows rather than legacy 81-row packing\',seed=42),indent=2),encoding=\'utf8\')\n records=[]\n for branch,grid in grids.items():\n  for kind in [\'RF\',\'1DCNN\',\'2DCNN\']:\n   folder=OUT/(branch+\'_\'+kind);folder.mkdir(exist_ok=True);oof=np.full(len(y),np.nan)\n   for k,(tr,te) in enumerate(plans):\n    f=folder/f\'fold_{k+1}\';f.mkdir(exist_ok=True);cache=f/\'heldout.csv\'\n    if cache.exists():pr=pd.read_csv(cache).score.to_numpy()\n    else:\n     print(branch,kind,k+1,\'training\',len(tr),\'test\',len(te),flush=True)\n     m=fit(kind,branch,grid,xy[tr[y[tr]==1]],xy[tr[y[tr]==0]],f,0);pr=predict(m,kind,grid,xy[te]);del m;gc.collect()\n     pd.DataFrame(dict(site=te,label=y[te],score=pr)).to_csv(cache,index=False)\n    oof[te]=pr;yy=y[te];hat=pr>=.5;p,r,_=precision_recall_curve(yy,pr)\n    rec=dict(branch=branch,model=kind,fold=k+1,train_positive=int(y[tr].sum()),train_negative=int((y[tr]==0).sum()),test_positive=int(yy.sum()),test_negative=int((yy==0).sum()),AUC=roc_auc_score(yy,pr),AP=average_precision_score(yy,pr),PR_AUC=auc(r,p),accuracy=accuracy_score(yy,hat),balanced_accuracy=balanced_accuracy_score(yy,hat),precision=precision_score(yy,hat,zero_division=0),recall=recall_score(yy,hat),F1=f1_score(yy,hat,zero_division=0));records.append(rec);pd.DataFrame(records).to_csv(OUT/\'fold_metrics.csv\',index=False)\n   pd.DataFrame(xy,columns=[\'x\',\'y\']).assign(label=y,fold=folds,score=oof).to_csv(folder/\'OOF_scores.csv\',index=False)\n pd.DataFrame(records).groupby([\'branch\',\'model\'])[[\'AUC\',\'AP\',\'PR_AUC\',\'accuracy\',\'balanced_accuracy\',\'precision\',\'recall\',\'F1\']].agg([\'mean\',\'std\']).to_csv(OUT/\'metrics_mean_std.csv\')\n (OUT/\'TRAINING_COMPLETE.json\').write_text(\'{"fold_models":30}\')\nif __name__==\'__main__\':main()\n', 'render_fixed_labels_figure7': '"""Adapt fixed-label spatial OOF to the supplied PAC_publication_four layout."""\nfrom pathlib import Path\nimport importlib.util, os\nimport numpy as np,pandas as pd\nif not hasattr(np,"trapezoid"): np.trapezoid=np.trapz\nfrom sklearn.metrics import roc_curve,roc_auc_score,average_precision_score\nR=Path(os.environ[\'CAOB_SPATIAL\']);S=R/\'figure_input\';S.mkdir(exist_ok=True)\nassert (R/\'TRAINING_COMPLETE.json\').exists()\nnames=[\'original-RF\',\'PCA-RF\',\'original-1DCNN\',\'PCA-1DCNN\',\'original-2DCNN\',\'PCA-transformed 2DCNN\'];pairs=[(\'Raw\',\'RF\'),(\'PCA\',\'RF\'),(\'Raw\',\'1DCNN\'),(\'PCA\',\'1DCNN\'),(\'Raw\',\'2DCNN\'),(\'PCA\',\'2DCNN\')];audit=[]\nfor name,(branch,model) in zip(names,pairs):\n d=pd.read_csv(R/(branch+\'_\'+model)/\'OOF_scores.csv\');y=d.label.to_numpy(int);s=d.score.to_numpy();d.rename(columns={\'label\':\'Label\'}).to_csv(S/(name+\'_evaluation_samples.csv\'),index=False)\n fp,tp,th=roc_curve(y,s,drop_intermediate=False);ntp=tp*y.sum();nfp=fp*(len(y)-y.sum());precision=np.divide(ntp,ntp+nfp,out=np.ones(len(tp)),where=(ntp+nfp)>0)\n pd.DataFrame(dict(recall=tp,FPR=fp,precision=precision,evaluation_unit_fraction=(ntp+nfp)/len(y))).to_csv(S/(name+\'_curves.csv\'),index=False)\n audit.append(dict(model=name,ROC_reproduced=True,archived_ROC_AUC=roc_auc_score(y,s),AP=average_precision_score(y,s)))\npd.DataFrame(audit).to_csv(S/\'metrics_and_ROC_audit.csv\',index=False)\nsource=Path(__file__).parent/\'PAC_publication_four.py\'\nspec=importlib.util.spec_from_file_location(\'publication\',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)\n# Use the supplied layout, while making all figure annotations fold mean ± SD.\ntext=source.read_text(encoding=\'utf-8-sig\')\ntext=text.replace("annotations=\'\\\\n\'.join(f\'{label}: {row[field]:.4f}\' for label,row in zip(LABELS,rows))", "stats=pd.read_csv(source.parent/\'fold_metrics.csv\'); metric={\'ROC\':\'AUC\',\'PR_AUC\':\'PR_AUC\',\'Balanced_accuracy\':\'balanced_accuracy\'}[kind]; annotations=\'\\\\n\'.join(f\'{label}: {stats.loc[(stats.branch==branch)&(stats.model==model),metric].mean():.4f} ± {stats.loc[(stats.branch==branch)&(stats.model==model),metric].std():.4f}\' for label,(branch,model) in zip(LABELS,[(\'Raw\',\'RF\'),(\'PCA\',\'RF\'),(\'Raw\',\'1DCNN\'),(\'PCA\',\'1DCNN\'),(\'Raw\',\'2DCNN\'),(\'PCA\',\'2DCNN\')])); heading += \' (fold mean ± SD)\'")\n# Execute a local adapted copy: original file remains unchanged.\nadapt=R/\'PAC_publication_four_spatial.py\';adapt.write_text(text,encoding=\'utf8\');spec=importlib.util.spec_from_file_location(\'spatial_publication\',adapt);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.main(S,R/\'Figure7\')\n(R/\'Figure7\'/\'README.txt\').write_text(\'Curves: pooled spatial out-of-fold predictions only. Combined-panel numerical annotations: five-fold mean ± sample SD, not pooled curve area. Fixed original negative labels; 5-degree blocks and 100-km cell-to-cell exclusion. All models share complete same-label 9x9 evaluation windows; RF/1D window-mean probabilities. PCA frozen/transductive. Original architecture and fixed parameters; legacy row packing replaced by real spatial windows. See protocol.json and fold_metrics.csv.\',encoding=\'utf8\')\n\n', 'spatial_two_schemes': '"""Spatial evaluation and two negative-sampling schemes. No target-AUC selection.\nRun with Anaconda Python. Output never overwrites original source/data.\n"""\nimport os\nos.environ[\'TF_CPP_MIN_LOG_LEVEL\']=\'2\'\nimport argparse,json,sys,hashlib\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.neighbors import BallTree\nfrom sklearn.model_selection import StratifiedGroupKFold\nfrom sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,roc_curve\nimport matplotlib\nmatplotlib.use(\'Agg\')\nimport matplotlib.pyplot as plt\nimport joblib\nROOT=Path(os.environ[\'CAOB_PROJECT\'])\nRAW=[\'176Hf-177Hf\',\'176Yb-177Hf\',\'206Pb-238U\',\'207Pb-235U\',\'207Pb_206Pb\',\'208Pb-232Th\',\'Al2O3\',\'CaO\',\'FeOT\',\'K2O\',\'LOI\',\'Lu176-Hf177\',\'MgO\',\'MnO\',\'Na2O\',\'P2O5\',\'SiO\',\'TDM1\',\'TDM2\',\'Th-U\',\'TiO2\',\'Hf(0)\',\'Hf(t)\']\nPARAMS={\'Raw\':dict(n_estimators=329,min_samples_split=2,min_samples_leaf=1,max_depth=16,bootstrap=True),\'PCA\':dict(n_estimators=248,min_samples_split=2,min_samples_leaf=4,max_depth=9,bootstrap=False)}\ndef geo(xy):\n    return np.column_stack((35+xy[:,1]/563*25,55+xy[:,0]/1799*80))\ndef distance(a,b):\n    return BallTree(np.radians(geo(b)),metric=\'haversine\').query(np.radians(geo(a)))[0][:,0]*6371.0088\n\ndef balanced(pool,n,rng):\n    # Round robin over geographic cells, rather than area-proportional draws.\n    if len(pool)<n: raise ValueError(\'Insufficient candidates; no automatic buffer relaxation\')\n    ll=geo(pool);cell=np.floor(ll/3).astype(int)\n    groups={}\n    for i,key in enumerate(map(tuple,cell)):groups.setdefault(key,[]).append(i)\n    for v in groups.values():rng.shuffle(v)\n    keys=list(groups);rng.shuffle(keys);chosen=[]\n    while len(chosen)<n:\n        for k in keys:\n            if groups[k]:chosen.append(groups[k].pop())\n            if len(chosen)==n:break\n    return pool[chosen]\n\ndef patches(grid,xy):\n    return np.stack([grid[y-4:y+5,x-4:x+5] for x,y in xy])\n\ndef pub(pool,pos,grid,rng,bags):\n    # Out-of-bag PU scores: never score a U point with a model trained on it.\n    u=grid[pool[:,1],pool[:,0]];p=grid[pos[:,1],pos[:,0]]\n    sums=np.zeros(len(pool));counts=np.zeros(len(pool),int)\n    for b in range(bags):\n        ix=rng.choice(len(pool),len(pos),replace=False)\n        model=RandomForestClassifier(**PARAMS[\'Raw\'],random_state=4200+b,n_jobs=4)\n        model.fit(np.concatenate([p,u[ix]]),np.r_[np.ones(len(p)),np.zeros(len(ix))])\n        keep=np.ones(len(pool),bool);keep[ix]=False\n        sums[keep]+=model.predict_proba(u[keep])[:,1];counts[keep]+=1\n    if (counts==0).any():raise ValueError(\'Unscored PU candidates; increase bags\')\n    return sums/counts\n\ndef savefig(folder,name):\n    for ext in [\'svg\',\'jpg\']:plt.savefig(folder/(name+\'.\'+ext),bbox_inches=\'tight\',dpi=180)\n    plt.close()\n\ndef coordinates(folder,name,xy,**extra):\n    t=pd.DataFrame(xy,columns=[\'x\',\'y\']);ll=geo(xy)\n    t[\'Longitude\']=ll[:,1];t[\'Latitude\']=ll[:,0]\n    for k,v in extra.items():t[k]=v\n    t.to_csv(folder/(name+\'.csv\'),index=False,encoding=\'utf-8-sig\')\n\ndef fit(kind,branch,grid,pos,neg,folder,epochs):\n    y=np.r_[np.ones(len(pos)),np.zeros(len(neg))].astype(int)\n    xy=np.concatenate([pos,neg]);p=patches(grid,xy)\n    if kind==\'RF\':\n        m=RandomForestClassifier(**PARAMS[branch],random_state=0,n_jobs=4)\n        m.fit(p.reshape(-1,23),np.repeat(y,81));joblib.dump(m,folder/\'model.joblib\');return m\n    import tensorflow as tf\n    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42)\n    L=tf.keras.layers\n    if kind==\'1DCNN\':\n        inp=L.Input((23,1));z=L.Conv1D(64,3,padding=\'same\',activation=\'relu\')(inp);z=L.BatchNormalization()(z);z=L.Conv1D(32,3,padding=\'same\',activation=\'relu\')(z);z=L.BatchNormalization()(z);z=L.Flatten()(z)\n        X=p.reshape(-1,23,1);y=np.repeat(y,81)\n    else:\n        inp=L.Input((9,9,23));z=L.Conv2D(64,3,padding=\'same\',activation=\'relu\')(inp);z=L.BatchNormalization()(z);z=L.Conv2D(32,3,padding=\'same\',activation=\'relu\')(z);z=L.BatchNormalization()(z);z=L.GlobalAveragePooling2D()(z);X=p\n    m=tf.keras.Model(inp,L.Dense(1,activation=\'sigmoid\')(z));m.compile(optimizer=\'adam\',loss=\'binary_crossentropy\',metrics=[\'accuracy\'])\n    patience=5 if branch==\'Raw\' and kind==\'1DCNN\' else 3\n    cb=[] if branch==\'PCA\' and kind==\'1DCNN\' else [tf.keras.callbacks.EarlyStopping(monitor=\'loss\',patience=patience,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor=\'loss\',factor=.1,patience=patience,min_lr=1e-6)]\n    ne=11 if branch==\'PCA\' and kind==\'1DCNN\' else 100\n    h=m.fit(X,y,batch_size=32,epochs=epochs or ne,callbacks=cb,verbose=0)\n    pd.DataFrame(h.history).to_csv(folder/\'training_history.csv\',index=False);m.save(folder/\'model.keras\');return m\n\ndef predict(m,kind,grid,xy):\n    out=[]\n    for start in range(0,len(xy),512):\n        p=patches(grid,xy[start:start+512])\n        if kind==\'RF\':s=m.predict_proba(p.reshape(-1,23))[:,1].reshape(-1,81).mean(1)\n        elif kind==\'1DCNN\':s=m.predict(p.reshape(-1,23,1),batch_size=8192,verbose=0).reshape(-1,81).mean(1)\n        else:s=m.predict(p,batch_size=256,verbose=0).ravel()\n        out.extend(s)\n    return np.array(out)\n\ndef main():\n    a=argparse.ArgumentParser();a.add_argument(\'--output\',default=str(ROOT/\'fig/New_negative_sample\'));a.add_argument(\'--folds\',type=int,default=5);a.add_argument(\'--exclusion-km\',type=float,default=50);a.add_argument(\'--test-buffer-km\',type=float,default=100);a.add_argument(\'--block-deg\',type=float,default=5);a.add_argument(\'--bags\',type=int,default=20);a.add_argument(\'--epochs\',type=int,default=0);a.add_argument(\'--prepare-only\',action=\'store_true\');a=a.parse_args()\n    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)\n    rng=np.random.default_rng(20261117)\n    grids={}\n    for branch,rel,cols in [(\'Raw\',\'original_analyses/grayscale_data_combined_test.csv\',RAW),(\'PCA\',\'data/PCA_with_labels_Test.csv\',[f\'PCA{i}\' for i in range(1,24)])]:\n        df=pd.read_csv(ROOT/\'data\'/rel).sort_values([\'y\',\'x\'])\n        expected=np.column_stack((np.tile(np.arange(1800),564),np.repeat(np.arange(564),1800)))\n        if not np.array_equal(df[[\'x\',\'y\']].values,expected):raise ValueError(\'Grid coordinate convention differs\')\n        v=df[cols].to_numpy(np.float32)\n        if not np.isfinite(v).all():raise ValueError(\'Missing/nonfinite evidence: explicit handling required\')\n        grids[branch]=v.reshape(564,1800,23)\n    loc=pd.read_excel(ROOT/\'data/data/Sample Location_delet_orogenic_gold.xlsx\')\n    xy=np.rint(np.column_stack(((loc.Longitude-55)/80*1799,(loc.Latitude-35)/25*563))).astype(int)\n    valid=(xy[:,0]>=4)&(xy[:,0]<1796)&(xy[:,1]>=4)&(xy[:,1]<560)\n    loc.assign(x=xy[:,0],y=xy[:,1],valid_patch=valid).to_csv(out/\'deposit_coordinate_audit.csv\',index=False)\n    pos=np.unique(xy[valid],axis=0)\n    # Fixed candidate pool and seed. All sites are unlabeled, not confirmed barren.\n    xx,yy=np.meshgrid(np.arange(8,1792,9),np.arange(8,556,9));pool=np.column_stack([xx.ravel(),yy.ravel()])\n    pool=pool[distance(pool,xy)>=a.exclusion_km]\n    negative=balanced(pool,len(pos),rng)\n    coordinates(out,\'scheme_A_negative_centers\',negative)\n    scores=pub(pool,pos,grids[\'Raw\'],rng,a.bags)\n    low=pool[scores<=np.quantile(scores,.25)]\n    negative_b=balanced(low,len(pos),rng)\n    coordinates(out,\'scheme_B_PUB_pseudo_negative_centers\',negative_b)\n    coordinates(out,\'PUB_candidate_scores\',pool,pu_oob_score=scores)\n    allxy=np.concatenate([pos,negative]);labels=np.r_[np.ones(len(pos)),np.zeros(len(negative))].astype(int)\n    block=np.floor(geo(allxy)/a.block_deg).astype(int);groups=np.unique(block,axis=0,return_inverse=True)[1]\n    splits=list(StratifiedGroupKFold(a.folds,shuffle=True,random_state=42).split(allxy,labels,groups))\n    assign=np.zeros(len(labels),int)\n    for i,(_,te) in enumerate(splits):assign[te]=i\n    coordinates(out,\'fixed_evaluation_sites\',allxy,label=labels,fold=assign,block=groups)\n    protocol=dict(vars(a),seed=20261117,valid_unique_deposits=len(pos),excluded_deposit_records=int((~valid).sum()),geological_mask=False,coverage_mask=False,negative_status=\'unlabeled/pseudo-negative\',PUB=\'RF bagging with OOB scoring; lower quartile + geographic balance is an adaptation, not verbatim paper method\',PCA=\'Frozen original full-grid PCA: transductive covariates; no held-out labels fitted\',evaluation=\'Same positive/background sites across schemes. 9x9 patch scores; RF/1D use patch-mean score; test label 0 means background, not confirmed absence\',buffer=\'Train patch centers must exceed test buffer plus 2 patch radii; all train centers in held-out geographic blocks excluded\',parameters=PARAMS)\n    (out/\'protocol.json\').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding=\'utf8\')\n    if a.prepare_only:return\n    records=[];maps={}\n    map_x,map_y=np.meshgrid(np.arange(4,1796,9),np.arange(4,560,9));mapxy=np.column_stack([map_x.ravel(),map_y.ravel()]);area=np.cos(np.radians(geo(mapxy)[:,0]))\n    for scheme in [\'A\',\'B_PUB\']:\n        fold_training=[]\n        for i,(_,te) in enumerate(splits):\n            test=allxy[te];held=set(groups[te]);pb=np.floor(geo(pos)/a.block_deg).astype(int)\n            held_cells=set(map(tuple,block[te]));pkeep=np.array([tuple(k) not in held_cells for k in pb])\n            # Conservative extra ~60 km avoids receptive-field overlap at this grid scale.\n            p=pos[pkeep & (distance(pos,test)>a.test_buffer_km+60)]\n            ub=np.floor(geo(pool)/a.block_deg).astype(int)\n            u=pool[np.array([tuple(k) not in held_cells for k in ub]) & (distance(pool,test)>a.test_buffer_km+60)]\n            if scheme==\'B_PUB\':\n                s=pub(u,p,grids[\'Raw\'],np.random.default_rng(6100+i),a.bags);u=u[s<=np.quantile(s,.25)]\n            n=balanced(u,len(p),np.random.default_rng(7100+i))\n            if len(p)<2 or len(np.unique(labels[te]))!=2:raise ValueError(\'Degenerate spatial fold; revise design explicitly\')\n            fold_training.append((p,n,te))\n        for branch,grid in grids.items():\n            for kind in [\'RF\',\'1DCNN\',\'2DCNN\']:\n                folder=out/scheme/(branch+\'_\'+kind);folder.mkdir(parents=True,exist_ok=True)\n                oof=np.full(len(labels),np.nan)\n                for i,(p,n,te) in enumerate(fold_training):\n                    f=folder/f\'fold_{i+1}\';f.mkdir(exist_ok=True)\n                    coordinates(f,\'train_centers\',np.concatenate([p,n]),label=np.r_[np.ones(len(p)),np.zeros(len(n))])\n                    print(scheme,branch,kind,\'fold\',i+1,\'train\',len(p),len(n),flush=True)\n                    m=fit(kind,branch,grid,p,n,f,a.epochs);pr=predict(m,kind,grid,allxy[te]);oof[te]=pr;y=labels[te];hat=pr>=.5\n                    r=dict(scheme=scheme,branch=branch,model=kind,fold=i+1,train_positive=len(p),train_background=len(n),test_positive=int(y.sum()),test_background=int((y==0).sum()),AUC=roc_auc_score(y,pr),AP=average_precision_score(y,pr),accuracy=accuracy_score(y,hat),balanced_accuracy=balanced_accuracy_score(y,hat),precision=precision_score(y,hat,zero_division=0),recall=recall_score(y,hat),F1=f1_score(y,hat,zero_division=0));records.append(r)\n                    pd.DataFrame(records).to_csv(out/\'fold_metrics.csv\',index=False)\n                    del m\n                coordinates(folder,\'OOF_scores\',allxy,label=labels,fold=assign,score=oof)\n                fpr,tpr,_=roc_curve(labels,oof);plt.figure();plt.plot(fpr,tpr);plt.plot([0,1],[0,1],\'--\');plt.xlabel(\'False positive rate (background)\');plt.ylabel(\'True positive rate\');savefig(folder,\'spatial_OOF_ROC\')\n                final=folder/\'final\';final.mkdir(exist_ok=True);n=negative if scheme==\'A\' else negative_b\n                m=fit(kind,branch,grid,pos,n,final,a.epochs);pr=predict(m,kind,grid,mapxy);maps[(scheme,branch,kind)]=pr\n                coordinates(folder,\'full_map_patch_scores\',mapxy,score=pr)\n                plt.figure(figsize=(15,5));plt.scatter(geo(mapxy)[:,1],geo(mapxy)[:,0],c=pr,s=4,vmin=0,vmax=1,cmap=\'coolwarm\',marker=\'s\',rasterized=True);plt.colorbar(label=\'Prospectivity score\');plt.xlabel(\'Longitude\');plt.ylabel(\'Latitude\');savefig(folder,\'prospectivity_map\')\n                pp=predict(m,kind,grid,pos);rows=[]\n                for t in np.linspace(0,1,101):rows.append(dict(threshold=t,area_fraction=float(area[pr>=t].sum()/area.sum()),deposit_capture=float((pp>=t).mean())))\n                pa=pd.DataFrame(rows);pa.to_csv(folder/\'PA_descriptive.csv\',index=False);plt.figure();plt.plot(pa.area_fraction,pa.deposit_capture);plt.xlabel(\'Latitude-weighted mapped area fraction\');plt.ylabel(\'Training deposit capture (descriptive)\');savefig(folder,\'PA_descriptive\');del m\n    metrics=pd.DataFrame(records);cols=[\'AUC\',\'AP\',\'accuracy\',\'balanced_accuracy\',\'precision\',\'recall\',\'F1\'];metrics.groupby([\'scheme\',\'branch\',\'model\'])[cols].agg([\'mean\',\'std\']).to_csv(out/\'metrics_mean_std.csv\')\n    from scipy.stats import spearmanr\n    compare=[]\n    for branch in grids:\n        for kind in [\'RF\',\'1DCNN\',\'2DCNN\']:\n            x=maps[\'A\',branch,kind];y=maps[\'B_PUB\',branch,kind];ix=np.argsort(x)[-int(len(x)*.1):];iy=np.argsort(y)[-int(len(y)*.1):];compare.append(dict(branch=branch,model=kind,spearman=spearmanr(x,y).statistic,top10_jaccard=len(set(ix)&set(iy))/len(set(ix)|set(iy)),mean_absolute_difference=np.abs(x-y).mean()))\n    pd.DataFrame(compare).to_csv(out/\'map_stability.csv\',index=False)\n    (out/\'COMPLETE.json\').write_text(json.dumps(dict(status=\'complete\',models=12,folds=a.folds)),encoding=\'utf8\')\nif __name__==\'__main__\':main()\n\n\n', 'RF_spatial_block_CV': "import os\nimport argparse, json\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams.update({'font.family':'Times New Roman','axes.unicode_minus':False,'svg.fonttype':'none','mathtext.fontset':'stix'})\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, ParameterSampler\nfrom sklearn.metrics import roc_curve, roc_auc_score\nimport joblib\n\nROOT=Path(os.environ['CAOB_PROJECT'])/'data'\n\ndef splits(X,y,xy,groups,k,buffer):\n    result=[]\n    for tr,va in StratifiedGroupKFold(k,shuffle=True,random_state=42).split(X,y,groups):\n        if buffer>0:\n            from scipy.spatial import cKDTree\n            distance=cKDTree(xy[va]).query(xy[tr],k=1)[0]\n            tr=tr[distance>buffer]\n        if len(np.unique(y[tr]))!=2 or len(np.unique(y[va]))!=2:\n            raise ValueError('Spatial fold has one class; revise block size/folds using sampling design, not AUC.')\n        assert not set(groups[tr]) & set(groups[va])\n        result.append((tr,va))\n    return result\n\ndef savefig(out,name):\n    dpi = 100 if name == 'prediction_full_map' else 300\n    for ext in ['jpg','svg']: plt.savefig(out/f'{name}.{ext}',dpi=dpi,bbox_inches='tight')\n    plt.close()\n\ndef export_parameter_text(out, kind, model, selected):\n    lines = [kind + ' — spatial block cross-validation for parameter tuning',\n             'Final mapping model: selected by inner spatial CV on all labelled training samples.',\n             'Selection metric: ROC AUC; 10 sampled candidates by default.',\n             '', 'Selected hyperparameters:']\n    lines += [f'{key} = {value}' for key, value in sorted(selected.items())]\n    lines += ['', 'Complete fitted estimator parameters (includes fixed/default values):']\n    lines += [f'{key} = {value}' for key, value in sorted(model.get_params().items())]\n    lines += ['', 'Spatial configuration and outer evaluation: see metrics.json.',\n              'Outer-fold selected parameters are recorded separately in fold_AUC.csv.']\n    (out / 'RF_best_params.txt').write_text('\\n'.join(lines) + '\\n', encoding='utf-8-sig')\n\ndef plot_roc(out, kind, fpr, tpr, score):\n    plt.figure(figsize=(7,6))\n    plt.plot(fpr,tpr,linewidth=2,label=f'{kind}, spatial OOF AUC = {score:.3f}')\n    plt.plot([0,1],[0,1],linestyle='--',linewidth=1,label='Random prediction')\n    plt.xlabel('False positive rate',fontsize=14);plt.ylabel('True positive rate',fontsize=14)\n    plt.title(f'ROC Curve of the {kind} Model',fontsize=16)\n    plt.xticks(fontsize=12);plt.yticks(fontsize=12);plt.legend(fontsize=12);plt.grid(True)\n    savefig(out,'ROC')\n\ndef plot_map(out, kind, full):\n    plt.figure(figsize=(18,5.64),dpi=100)\n    plt.tricontourf(full.x,full.y,full.probability,levels=14,cmap='coolwarm')\n    plt.colorbar(label='Predicted Probability');plt.axis('off')\n    plt.title('Predicted Probability Contour Map - '+kind)\n    savefig(out,'prediction_full_map')\n\ndef plot_pa(out, kind, pa):\n    frame=pa.sort_values('lower',ascending=False)\n    x=np.arange(len(frame));width=.38\n    labels=[f'{v:.2f}-{v+.1:.2f}' for v in frame.lower]\n    plt.figure(figsize=(10,6))\n    plt.bar(x-width/2,frame.area_percent,width,label=kind+' predicted area (%)',edgecolor='black',color='tab:blue')\n    plt.bar(x+width/2,frame.positive_capture_percent,width,label=kind+' captured positive samples (%)',edgecolor='black',color='tab:orange')\n    plt.xlabel('Prospectivity score class',fontsize=14);plt.ylabel('Percentage (%)',fontsize=14)\n    plt.title('Prediction-Area Plot of the '+kind+' Model (descriptive)',fontsize=16)\n    plt.xticks(x,labels,rotation=45,ha='right',fontsize=12);plt.yticks(fontsize=12)\n    plt.legend(fontsize=12);plt.grid(True,axis='y');plt.tight_layout();savefig(out,'PA_descriptive')\n\ndef run(a,kind):\n    out=Path(a.output)/kind; out.mkdir(parents=True,exist_ok=True)\n    folder=ROOT/('data' if kind=='PCA_RF' else 'original_analyses')\n    train=folder/('PCA_with_labels_Train.csv' if kind=='PCA_RF' else 'grayscale_data_combined_train.csv')\n    test=folder/('PCA_with_labels_Test.csv' if kind=='PCA_RF' else 'grayscale_data_combined_test.csv')\n    df=pd.read_csv(train)\n    features=[f'PCA{i}' for i in range(1,24)] if kind=='PCA_RF' else df.loc[:,'176Hf-177Hf':'Hf(t)'].columns.tolist()\n    df=df.loc[pd.to_numeric(df.Label,errors='coerce').isin([0,1])].copy().reset_index(drop=True)\n    if df.duplicated(['x','y']).any(): raise ValueError('Duplicate training coordinates must be resolved.')\n    X=np.nan_to_num(df[features].to_numpy(dtype='float32'),nan=0,posinf=0,neginf=0)\n    y=df.Label.to_numpy(dtype=int); xy=df[['x','y']].to_numpy(float)\n    if not np.isfinite(xy).all(): raise ValueError('Invalid coordinates')\n    blocks=np.floor(xy/a.block_size).astype(int)\n    groups=np.unique(blocks,axis=0,return_inverse=True)[1]\n    distribution={'n_estimators':np.arange(2,500),'max_depth':np.arange(2,31),'min_samples_split':[2],'min_samples_leaf':[1,2,4],'bootstrap':[True,False]}\n    # Fixed random candidates; evaluation itself uses spatial folds, never random pixel folds.\n    candidates=[{k:[v] for k,v in p.items()} for p in ParameterSampler(distribution,n_iter=a.candidates,random_state=0)]\n    def tune(xx,yy,coords,gg):\n        cv=splits(xx,yy,coords,gg,a.inner_folds,a.buffer)\n        search=GridSearchCV(RandomForestClassifier(random_state=0,n_jobs=1),candidates,scoring='roc_auc',cv=cv,n_jobs=a.jobs,error_score='raise')\n        search.fit(xx,yy)\n        return search\n    prediction=np.full(len(y),np.nan); fold=np.zeros(len(y),int); rows=[]\n    for i,(tr,va) in enumerate(splits(X,y,xy,groups,a.outer_folds,a.buffer),1):\n        print(kind,'outer fold',i,flush=True)\n        search=tune(X[tr],y[tr],xy[tr],groups[tr]); prediction[va]=search.predict_proba(X[va])[:,1]; fold[va]=i\n        pd.DataFrame(search.cv_results_).to_csv(out/f'tuning_outer_{i}.csv',index=False)\n        rows.append({'fold':i,'auc':roc_auc_score(y[va],prediction[va]),'n_train':len(tr),'n_valid':len(va),'best_params':search.best_params_})\n    df.assign(block=groups,fold=fold,oof_probability=prediction).to_csv(out/'spatial_OOF_predictions.csv',index=False)\n    pd.DataFrame(rows).to_csv(out/'fold_AUC.csv',index=False)\n    fpr,tpr,threshold=roc_curve(y,prediction); score=roc_auc_score(y,prediction)\n    pd.DataFrame({'fpr':fpr,'tpr':tpr,'threshold':threshold}).to_csv(out/'ROC_data.csv',index=False)\n    plot_roc(out,kind,fpr,tpr,score)\n    plt.scatter(xy[:,0],xy[:,1],c=fold,s=3,cmap='tab10');plt.gca().invert_yaxis();plt.colorbar(label='Outer fold');savefig(out,'spatial_folds')\n    final=tune(X,y,xy,groups);joblib.dump(final.best_estimator_,out/'RF_model.joblib');pd.DataFrame(final.cv_results_).to_csv(out/'final_tuning.csv',index=False)\n    export_parameter_text(out,kind,final.best_estimator_,final.best_params_)\n    pd.DataFrame({'feature':features,'importance':final.best_estimator_.feature_importances_}).to_csv(out/'feature_importance.csv',index=False)\n    # Predict entire supplied map in chunks, without using its labels as independent validation.\n    mapped=[]\n    for chunk in pd.read_csv(test,chunksize=100000):\n        xx=np.nan_to_num(chunk[features].to_numpy(dtype='float32'),nan=0,posinf=0,neginf=0)\n        chunk=chunk[[c for c in ['x','y','Label'] if c in chunk]].copy();chunk['probability']=final.predict_proba(xx)[:,1];mapped.append(chunk)\n    full=pd.concat(mapped,ignore_index=True);full.to_csv(out/'full_map_predictions.csv',index=False)\n    if full.duplicated(['x','y']).any(): raise ValueError('Full map contains duplicate coordinates')\n    grid=full.pivot(index='y',columns='x',values='probability')\n    plot_map(out,kind,full)\n    # P-A is descriptive: equal-area pixels; labels can overlap training locations.\n    positive=pd.to_numeric(full.Label,errors='coerce').eq(1).to_numpy(); p=full.probability.to_numpy();records=[]\n    for lo in np.arange(0,1,.1):\n        mask=(p>=lo)&((p<lo+.1) if lo<.9 else (p<=1));records.append({'lower':lo,'area_percent':mask.mean()*100,'positive_capture_percent':100*positive[mask].sum()/max(1,positive.sum())})\n    pa=pd.DataFrame(records);pa.to_csv(out/'PA_descriptive.csv',index=False);plot_pa(out,kind,pa)\n    (out/'metrics.json').write_text(json.dumps({'nested_spatial_oof_auc':score,'fold_auc_mean':float(np.mean([r['auc'] for r in rows])),'fold_auc_sd':float(np.std([r['auc'] for r in rows],ddof=1)),'best_params':final.best_params_,'configuration':vars(a),'features':features,'note':'Pixel blocks; block size requires spatial-range justification. PCA upstream fitted globally is not refitted here; OOF is conditional on supplied features. PA/full-map are descriptive, not independent validation.'},ensure_ascii=False,indent=2,default=lambda x:x.item()),encoding='utf8')\n\nif __name__=='__main__':\n    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['PCA_RF','Original_RF','both'],default='both');p.add_argument('--block-size',type=float,default=64);p.add_argument('--buffer',type=float,default=12);p.add_argument('--outer-folds',type=int,default=5);p.add_argument('--inner-folds',type=int,default=3);p.add_argument('--candidates',type=int,default=10);p.add_argument('--jobs',type=int,default=4);p.add_argument('--output',default=str(Path(__file__).parent/'RF_spatial_results'));a=p.parse_args()\n    if a.block_size<=0 or a.buffer<0: p.error('Invalid block/buffer size')\n    for kind in (['Original_RF','PCA_RF'] if a.mode=='both' else [a.mode]): run(a,kind)\n", 'caob_permutation_test': '"""Density maps and conditional deposit-location randomization at 80 E."""\nimport argparse\nfrom pathlib import Path\n\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nfrom matplotlib.colors import LogNorm\nfrom matplotlib.font_manager import FontProperties, findfont\nimport numpy as np\nimport pandas as pd\n\nARTICLE = Path("D:/homework/Article/2025/MPM/JGE/Supplementray Material")\nOUTPUT = Path("D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/fig/permutation_test")\nCOORDS = ["Longitude", "Latitude"]\n\n\ndef main():\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--tables", type=Path, default=ARTICLE)\n    parser.add_argument("--output", type=Path, default=OUTPUT)\n    parser.add_argument("--draws", type=int, default=19999)\n    args = parser.parse_args()\n    if args.draws < 1:\n        parser.error("--draws must be positive")\n\n    def table(number, columns, sheet=0):\n        return pd.read_excel(args.tables / f"Supplementary Material {number}.xlsx",\n                             sheet_name=sheet, header=1, usecols=columns)\n\n    audit = []\n\n    def clean(frame, name, keys=()):\n        values = frame.copy()\n        values[COORDS] = values[COORDS].apply(pd.to_numeric, errors="coerce")\n        valid = values.dropna(subset=COORDS)\n        region = valid.loc[valid.Longitude.between(55, 135)\n                           & valid.Latitude.between(35, 60)]\n        used = region.drop_duplicates(list(keys)) if keys else region\n        audit.append(dict(layer=name, input_rows=len(frame), numeric_coordinates=len(valid),\n                          inside_domain=len(region), retained=len(used)))\n        return used\n\n    gold = table(1, [*COORDS, "Deposit", "Type"])\n    gold = gold.loc[gold.Type.str.strip().str.casefold().eq("orogenic")].dropna(subset=["Deposit"])\n    gold["Deposit"] = gold.Deposit.str.strip().str.casefold()\n    gold = clean(gold, "gold", ["Deposit", *COORDS])\n    zircon = table(2, [*COORDS, "Unique Sample No."])\n    hf = table(2, COORDS, 1)\n    dyke = table(3, ["LATITUDE MIN", "LATITUDE MAX", "LONGITUDE MIN", "LONGITUDE MAX"])\n    for coordinate in COORDS:\n        endpoints = [f"{coordinate.upper()} {end}" for end in ("MIN", "MAX")]\n        dyke[coordinate] = dyke[endpoints].apply(pd.to_numeric, errors="coerce").mean(axis=1)\n    points = {\n        "Zircon U-Pb analyses": clean(zircon, "zircon_upb"),\n        "Zircon Hf analyses": clean(hf, "zircon_hf"),\n        "GEOROC dyke records": clean(dyke, "dyke"),\n        "Orogenic gold deposits": gold,\n    }\n    samples = clean(zircon.dropna(subset=["Unique Sample No."]), "zircon_upb_samples",\n                    ["Unique Sample No."])\n    args.output.mkdir(parents=True, exist_ok=True)\n    pd.DataFrame(audit).to_csv(args.output / "input_audit.csv", index=False)\n    findfont(FontProperties(family="Calibri", weight="bold"), fallback_to_default=False)\n    plt.rcParams.update({"svg.fonttype": "none", "font.family": "Calibri",\n                         "font.weight": "bold", "axes.labelweight": "bold",\n                         "axes.titleweight": "bold", "font.size": 10,\n                         "mathtext.fontset": "custom", "mathtext.default": "bf",\n                         "mathtext.rm": "Calibri:bold", "mathtext.bf": "Calibri:bold"})\n\n    def save(fig, name):\n        for extension in ("jpg", "svg"):\n            fig.savefig(args.output / f"{name}.{extension}", dpi=320, facecolor="white")\n        plt.close(fig)\n\n    results = []\n    rng = np.random.default_rng(20261120)\n    nulls = []\n    for step in (1, 2):\n        # Anchor at the tested meridian so no cell crosses the boundary.\n        x = np.unique(np.r_[55, np.arange(80, 55, -step), np.arange(80, 135, step), 135])\n        y = np.r_[np.arange(35, 60, step), 60]\n        lon, lat = np.meshgrid((x[1:] + x[:-1]) / 2, (y[1:] + y[:-1]) / 2)\n        area = 6371.0088**2 * np.outer(np.diff(np.sin(np.deg2rad(y))), np.diff(np.deg2rad(x)))\n\n        def count(frame):\n            return np.histogram2d(frame.Latitude, frame.Longitude, bins=(y, x))[0].astype(int)\n\n        layers = {label: count(frame) for label, frame in points.items()}\n        deposits = layers["Orogenic gold deposits"]\n        assert deposits.sum() == len(gold)\n        z = layers["Zircon U-Pb analyses"]\n        d = layers["GEOROC dyke records"]\n        s = count(samples)\n\n        def density_class(counts):\n            density = counts / area\n            positive = density[density > 0]\n            return np.where(density == 0, 0, np.where(density <= np.median(positive), 1, 2))\n\n        if step == 1:\n            fig, axes = plt.subplots(4, 1, figsize=(9, 9.6), layout="constrained")\n            for ax, (label, counts), color in zip(axes, layers.items(),\n                                                ["viridis", "cividis", "magma", "YlGnBu"]):\n                density = counts / area * 10000\n                positive = density[density > 0]\n                ax.set_facecolor("#eeeeee")\n                for spine in ax.spines.values():\n                    spine.set(visible=True, color="#334155", linewidth=0.85)\n                ax.tick_params(direction="out", length=3, color="#334155")\n                mesh = ax.pcolormesh(x, y, np.ma.masked_equal(density, 0), cmap=color,\n                                     norm=LogNorm(vmin=positive.min(), vmax=positive.max()))\n                fig.colorbar(mesh, ax=ax, pad=.02, label="Count per 10,000 km²")\n                ax.axvline(80, color="#ba3a31", linestyle="--", linewidth=1)\n                ax.set(title=f"{label} (n = {counts.sum():,})", ylabel="Latitude (°N)")\n            axes[-1].set_xlabel("Longitude (°E)")\n            save(fig, "01_density_maps")\n            grid = pd.DataFrame({"longitude": lon.ravel(), "latitude": lat.ravel(),\n                                 "area_km2": area.ravel(), "zircon_samples": s.ravel(),\n                                 **{k: v.ravel() for k, v in layers.items()}})\n            grid.to_csv(args.output / "grid_1deg.csv", index=False)\n\n        latitude_band = np.floor((lat - 35) / 5).astype(int)\n        for width in (2, 4, 6, 0):\n            west = (lon >= 80 - width) & (lon < 80)\n            east = (lon >= 80) & (lon < 80 + width)\n            gap = west | east\n            shoulders = ((lon >= 80 - 2 * width) & (lon < 80 + 2 * width)) & ~gap\n            if width == 0:\n                gap = lon < 80\n                shoulders = ~gap\n            local = gap | shoulders\n            # Conditional on the local total, a low central share is a density deficit\n            # relative to the two flanking bands. Preserve each stratum\'s observed total.\n            observed = int(deposits[gap].sum())\n            for model, strata in {\n                "Area only": np.zeros_like(latitude_band),\n                "Latitude + zircon analyses + dyke records":\n                    latitude_band * 9 + density_class(z) * 3 + density_class(d),\n                "Latitude + zircon samples + dyke records":\n                    latitude_band * 9 + density_class(s) * 3 + density_class(d),\n            }.items():\n                simulated = np.zeros(args.draws, dtype=int)\n                expected = 0.\n                movable = 0\n                for stratum in np.unique(strata[local]):\n                    match = local & (strata == stratum)\n                    n = int(deposits[match].sum())\n                    probability = area[match & gap].sum() / area[match].sum()\n                    simulated += rng.binomial(n, probability, size=args.draws)\n                    expected += n * probability\n                    movable += n if 0 < probability < 1 else 0\n                if width == 0:\n                    extreme = np.abs(simulated - expected) >= abs(observed - expected) - 1e-12\n                else:\n                    extreme = simulated <= observed\n                p = (1 + np.count_nonzero(extreme)) / (args.draws + 1)\n                results.append(dict(test="whole_domain_west_east" if width == 0 else "local_gap", grid_deg=step, half_width_deg=width, model=model,\n                                    observed_gap=observed, observed_shoulders=int(deposits[shoulders].sum()),\n                                    expected_gap=expected, exchangeable_deposits=movable,\n                                    gap_area_km2=area[gap].sum(), shoulder_area_km2=area[shoulders].sum(),\n                                    p_value=p, draws=args.draws))\n                if step == 1 and width == 2:\n                    nulls.append((model, simulated, observed, p))\n    pd.DataFrame(results).to_csv(args.output / "randomization_results.csv", index=False)\n    fig, axes = plt.subplots(1, len(nulls), figsize=(11, 3.8), layout="constrained",\n                             sharex=True, sharey=True)\n    fig.suptitle("Deposit-location randomization near 80°E", fontsize=15,\n                 fontweight="bold", color="#142b3b")\n    titles = ["Area-only null", "U–Pb analysis density", "U–Pb sample density"]\n    subtitles = ["Area-weighted allocation", "Also stratified by latitude and dyke density",\n                 "Also stratified by latitude and dyke density"]\n    maximum = max(max(null.max(), observed) for _, null, observed, _ in nulls)\n    counts = np.arange(maximum + 1)\n    for ax, title, subtitle, (_, null, observed, p) in zip(axes, titles, subtitles, nulls):\n        frequency = np.bincount(null, minlength=maximum + 1) / len(null) * 100\n        ax.bar(counts, frequency, width=.73, color="#40848b", zorder=3)\n        ax.axvline(observed, color="#ad492e", linestyle=(0, (4, 2)), linewidth=1.7,\n                   zorder=4)\n        ax.set_title(title, loc="left", fontsize=11, fontweight="bold",\n                     color="#142b3b", pad=17)\n        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=7.8, color="#486170")\n        ax.text(.98, .96, f"One-sided p = {p:.3f}", transform=ax.transAxes,\n                ha="right", va="top", fontsize=9, color="#142b3b",\n                bbox=dict(facecolor="white", edgecolor="none", alpha=.95, pad=3))\n        ax.set(xlim=(-.55, maximum + .55), xticks=counts)\n        ax.grid(axis="y", color="#dce5e8", linewidth=.7)\n        ax.tick_params(length=3, colors="#486170")\n        for spine in ax.spines.values():\n            spine.set(visible=True, color="#617884", linewidth=.8)\n    axes[0].set_ylabel("Share of randomizations (%)")\n    axes[0].plot([], [], color="#ad492e", linestyle=(0, (4, 2)), linewidth=1.7,\n                 label=f"Observed deposits = {observed}")\n    axes[0].legend(loc="upper left", frameon=False, fontsize=8)\n    fig.supxlabel("Number of deposits in 78–82°E", fontsize=10)\n    save(fig, "02_randomization_80E")\n\n\nif __name__ == "__main__":\n    main()\n\n\n\n\n\n\n', 'frozen_models': '# -*- coding: utf-8 -*-\n"""Post-analysis of42 trained models: current scores, supplied PA/four-panel methods, feature importance and CNN latent extraction. No fitting."""\nimport types\nENGINE_SOURCE=\'# -*- coding: utf-8 -*-\\n"""CAOB compatibility experiment: frozen A/B negatives, original positive labels,\\n18 fixed-setting refits, original-weight map reference, and full downstream analysis.\\nDefault runs training. --reuse-existing verifies and reuses the completed control weights.\\nAll ROC/P-A figures are descriptive, not independent spatial validation.\\nSingle file; requires original RAR, Revised JGE/data and installed scientific Python.\\n"""\\nFROZEN_NEGATIVE_CENTERS = {\\\'A\\\': [[917, 548], [1196, 242], [35, 17], [1025, 44], [1358, 431], [611, 467], [1430, 530], [1700, 224], [1565, 233], [1070, 386], [1160, 251], [773, 431], [1565, 53], [188, 350], [755, 503], [1763, 98], [683, 98], [953, 98], [1394, 287], [1610, 278], [1574, 17], [1376, 107], [566, 323], [170, 197], [1295, 17], [710, 359], [233, 26], [260, 323], [224, 548], [1574, 197], [908, 17], [863, 449], [1241, 305], [836, 62], [476, 134], [629, 107], [53, 17], [1088, 8], [332, 278], [746, 341], [1511, 8], [1367, 422], [161, 341], [566, 476], [530, 143], [728, 404], [1502, 440], [1034, 188], [242, 17], [980, 44], [737, 224], [593, 89], [197, 206], [8, 134], [791, 8], [1097, 305], [1358, 206], [1286, 494], [476, 440], [1475, 395], [1295, 62], [116, 368], [692, 8], [944, 350], [26, 188], [1700, 341], [53, 422], [314, 395], [845, 494], [548, 260], [962, 503], [989, 179], [944, 422], [1151, 494], [1043, 359], [746, 251], [1475, 503], [422, 98], [1709, 368], [1241, 8], [1412, 161], [1493, 62], [197, 440], [386, 188], [1790, 323], [107, 530], [359, 8], [998, 440], [485, 377], [8, 521], [206, 242], [1115, 197], [314, 503], [1772, 8], [134, 152], [611, 215], [1772, 197], [1583, 413], [863, 179], [881, 152], [1142, 359]], \\\'B\\\': [[80, 179], [35, 233], [287, 224], [593, 512], [1187, 8], [1034, 116], [467, 494], [35, 134], [98, 494], [1277, 17], [863, 287], [1709, 224], [179, 476], [1124, 485], [1673, 422], [71, 71], [125, 116], [674, 512], [170, 17], [197, 224], [1502, 26], [1655, 224], [845, 287], [188, 494], [161, 350], [1484, 8], [107, 116], [17, 485], [899, 467], [503, 503], [980, 278], [674, 17], [719, 233], [1061, 71], [251, 305], [188, 314], [431, 530], [53, 269], [152, 53], [179, 413], [269, 530], [1034, 17], [521, 422], [917, 341], [728, 134], [35, 323], [737, 224], [854, 296], [1115, 323], [260, 251], [26, 548], [1151, 485], [503, 305], [1034, 530], [98, 503], [989, 17], [143, 251], [1772, 512], [1718, 134], [197, 17], [701, 485], [35, 17], [1646, 251], [710, 98], [341, 251], [1664, 431], [1061, 8], [836, 53], [494, 422], [1673, 431], [1007, 323], [251, 431], [206, 269], [944, 467], [251, 422], [737, 71], [1268, 107], [701, 26], [62, 8], [989, 26], [1286, 44], [1133, 521], [44, 71], [1691, 359], [215, 539], [1259, 17], [1259, 233], [773, 8], [1655, 44], [1304, 350], [1682, 80], [1214, 530], [710, 224], [1043, 260], [1340, 8], [809, 17], [665, 422], [1259, 422], [1043, 35], [638, 422], [1196, 467]]}\\nPA_HELPER_SOURCE = \\\'"""Reanalyse saved predictions only. No model loading, fitting or PCA fitting."""\\\\nimport argparse,json,subprocess,hashlib\\\\nfrom pathlib import Path\\\\nimport numpy as np\\\\nimport pandas as pd\\\\nimport matplotlib\\\\nmatplotlib.use(\\\\\\\'Agg\\\\\\\')\\\\nimport matplotlib.pyplot as plt\\\\nplt.rcParams.update({\\\\\\\'font.family\\\\\\\':\\\\\\\'Times New Roman\\\\\\\',\\\\\\\'svg.fonttype\\\\\\\':\\\\\\\'none\\\\\\\'})\\\\nARCHIVE=\\\\\\\'C:/central asian mapping.rar\\\\\\\'\\\\nFILES={\\\\n\\\\\\\'Original_RF\\\\\\\':\\\\\\\'Revised/fig/original-RF_predictions.csv\\\\\\\',\\\\n\\\\\\\'PCA_RF\\\\\\\':\\\\\\\'Revised/fig/PCA-RF_predictions.csv\\\\\\\',\\\\n\\\\\\\'Original_1DCNN\\\\\\\':\\\\\\\'original_analyses/1DCNN/PCA_with_predictions_origin_1DCNN.csv\\\\\\\',\\\\n\\\\\\\'PCA_1DCNN\\\\\\\':\\\\\\\'Revised/fig/PCA-1DCNN_predictions.csv\\\\\\\',\\\\n\\\\\\\'Original_2DCNN\\\\\\\':\\\\\\\'Revised/fig/original-2DCNN_predictions.csv\\\\\\\',\\\\n\\\\\\\'PCA_2DCNN\\\\\\\':\\\\\\\'2DCNN_test/ensure/PCA_with_full_predictions_Test_2DCNN_test.csv\\\\\\\'}\\\\nROOT=Path(\\\\\\\'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data\\\\\\\')\\\\n\\\\ndef save(fig,out,name):\\\\n    for ext in [\\\\\\\'jpg\\\\\\\',\\\\\\\'svg\\\\\\\']:fig.savefig(out/(name+\\\\\\\'.\\\\\\\'+ext),dpi=300,bbox_inches=\\\\\\\'tight\\\\\\\')\\\\n    plt.close(fig)\\\\n\\\\ndef analyse(frame,points,out,name):\\\\n    frame=frame.rename(columns={\\\\\\\'X\\\\\\\':\\\\\\\'x\\\\\\\',\\\\\\\'Y\\\\\\\':\\\\\\\'y\\\\\\\'})\\\\n    candidates=[c for c in frame if \\\\\\\'probab\\\\\\\' in c.lower() or c==\\\\\\\'Predicted_Probability\\\\\\\']\\\\n    if len(candidates)!=1:raise ValueError(f\\\\\\\'{name}: ambiguous probability columns {candidates}\\\\\\\')\\\\n    col=candidates[0]\\\\n    frame[col]=pd.to_numeric(frame[col],errors=\\\\\\\'coerce\\\\\\\')\\\\n    if frame.duplicated([\\\\\\\'x\\\\\\\',\\\\\\\'y\\\\\\\']).any():raise ValueError(\\\\\\\'Duplicate pixel coordinates\\\\\\\')\\\\n    valid=np.isfinite(frame[col]);missing=int((~valid).sum());data=frame.loc[valid,[\\\\\\\'x\\\\\\\',\\\\\\\'y\\\\\\\',col]].copy()\\\\n    scores=data[col].to_numpy(); ss=np.sort(scores)\\\\n    if len(ss)==0:raise ValueError(\\\\\\\'No finite predictions\\\\\\\')\\\\n    if ss.min()<0 or ss.max()>1:raise ValueError(\\\\\\\'Scores outside probability interval\\\\\\\')\\\\n    lookup=data.set_index([\\\\\\\'x\\\\\\\',\\\\\\\'y\\\\\\\'])[col]\\\\n    deposits=points.copy();deposits[\\\\\\\'x\\\\\\\']=((deposits.Longitude-55)/80*1799).astype(int);deposits[\\\\\\\'y\\\\\\\']=((deposits.Latitude-35)/25*563).astype(int)\\\\n    deposits[\\\\\\\'probability\\\\\\\']=[lookup.get((x,y),np.nan) for x,y in zip(deposits.x,deposits.y)]\\\\n    deposits.to_csv(out/\\\\\\\'deposit_scores.csv\\\\\\\',index=False)\\\\n    ds=np.sort(deposits.probability.dropna().to_numpy())\\\\n    if not len(ds):raise ValueError(\\\\\\\'No deposits have predictions\\\\\\\')\\\\n    # All observed score cutoffs, with ties kept together; no arbitrary 0.1 probability bins.\\\\n    thresholds=np.unique(np.r_[0.,scores,np.nextafter(scores.max(),np.inf)])\\\\n    area=(len(ss)-np.searchsorted(ss,thresholds,side=\\\\\\\'left\\\\\\\'))/len(ss)*100\\\\n    capture=(len(ds)-np.searchsorted(ds,thresholds,side=\\\\\\\'left\\\\\\\'))/len(ds)*100\\\\n    density=np.divide(capture,area,out=np.full(len(area),np.nan),where=area>0)\\\\n    curve=pd.DataFrame({\\\\\\\'threshold\\\\\\\':thresholds,\\\\\\\'occupied_area_percent\\\\\\\':area,\\\\\\\'deposit_capture_percent\\\\\\\':capture,\\\\\\\'normalized_density\\\\\\\':density})\\\\n    curve.to_csv(out/\\\\\\\'PA_all_thresholds.csv\\\\\\\',index=False)\\\\n    # Reversed right axis: visual intersection satisfies Pr = 100 - Oa.\\\\n    delta=capture+area-100\\\\n    idx=np.flatnonzero(delta>=0)[-1];j=min(idx+1,len(delta)-1)\\\\n    weight=0 if delta[idx]==delta[j] else delta[idx]/(delta[idx]-delta[j])\\\\n    cross_t=thresholds[idx]+weight*(thresholds[j]-thresholds[idx])\\\\n    pr=capture[idx]+weight*(capture[j]-capture[idx]);oa=area[idx]+weight*(area[j]-area[idx])\\\\n    actual_a=float((scores>=cross_t).mean()*100);actual_p=float((ds>=cross_t).mean()*100)\\\\n    row=dict(model=name,intersection_threshold_interpolated=cross_t,prediction_rate_interpolated=pr,occupied_area_interpolated=oa,normalized_density_interpolated=pr/oa if oa else None,actual_threshold_area_percent=actual_a,actual_threshold_deposit_capture_percent=actual_p,actual_normalized_density=actual_p/actual_a if actual_a else None,total_map_rows=len(frame),finite_map_pixels=len(scores),missing_prediction_pixels=missing,total_deposits=len(deposits),deposits_with_prediction=len(ds),deposits_missing_prediction=len(deposits)-len(ds))\\\\n    fig,ax=plt.subplots(figsize=(10,6));right=ax.twinx()\\\\n    ax.plot(thresholds,capture,color=\\\\\\\'tab:blue\\\\\\\',label=\\\\\\\'Deposit capture\\\\\\\');right.plot(thresholds,area,color=\\\\\\\'tab:orange\\\\\\\',label=\\\\\\\'Occupied area\\\\\\\')\\\\n    ax.set_ylim(0,100);right.set_ylim(100,0);ax.set_xlim(0,1)\\\\n    ax.set_xlabel(\\\\\\\'Prospectivity threshold (score >= threshold)\\\\\\\');ax.set_ylabel(\\\\\\\'Prediction / success rate (%)\\\\\\\');right.set_ylabel(\\\\\\\'Occupied area (%) — reversed\\\\\\\')\\\\n    ax.axvline(cross_t,color=\\\\\\\'gray\\\\\\\',ls=\\\\\\\'--\\\\\\\');ax.set_title(name+\\\\\\\' P–A (known-deposit success assessment)\\\\\\\')\\\\n    ax.legend(loc=\\\\\\\'upper left\\\\\\\');right.legend(loc=\\\\\\\'upper right\\\\\\\');save(fig,out,\\\\\\\'PA\\\\\\\')\\\\n    fig,ax=plt.subplots(figsize=(7,6));ax.plot(area[::-1],capture[::-1],color=\\\\\\\'tab:blue\\\\\\\');ax.plot([0,100],[0,100],\\\\\\\'--\\\\\\\',color=\\\\\\\'gray\\\\\\\');ax.set(xlabel=\\\\\\\'Occupied area (%)\\\\\\\',ylabel=\\\\\\\'Known deposits captured (%)\\\\\\\',title=name+\\\\\\\' success-rate curve\\\\\\\',xlim=(0,100),ylim=(0,100));save(fig,out,\\\\\\\'success_rate\\\\\\\')\\\\n    (out/\\\\\\\'metrics.txt\\\\\\\').write_text(\\\\\\\'\\\\\\\\n\\\\\\\'.join(f\\\\\\\'{k} = {v}\\\\\\\' for k,v in row.items())+\\\\\\\'\\\\\\\\nDescriptive assessment using known deposits, not independent validation.\\\\\\\\nArea fractions assume equal-area pixels. Missing predictions are excluded, not filled.\\\\\\\\nThe interpolated crossing may not be exactly attainable with discrete/tied scores.\\\\\\\\n\\\\\\\',encoding=\\\\\\\'utf-8-sig\\\\\\\')\\\\n    return row\\\\n\\\\ndef main():\\\\n    p=argparse.ArgumentParser();p.add_argument(\\\\\\\'--output\\\\\\\',default=\\\\\\\'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/fig/P-C\\\\\\\');p.add_argument(\\\\\\\'--archive\\\\\\\',default=ARCHIVE);a=p.parse_args()\\\\n    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)\\\\n    points=pd.read_excel(ROOT/\\\\\\\'data/Sample Location_delet_orogenic_gold.xlsx\\\\\\\')\\\\n    rows=[];manifest=[]\\\\n    for name,member in FILES.items():\\\\n        out=root/name;out.mkdir(exist_ok=True);cached=out/\\\\\\\'saved_predictions.csv\\\\\\\'\\\\n        with cached.open(\\\\\\\'wb\\\\\\\') as f:\\\\n            subprocess.run([\\\\\\\'tar\\\\\\\',\\\\\\\'-xOf\\\\\\\',a.archive,\\\\\\\'central asian mapping/\\\\\\\'+member],stdout=f,check=True)\\\\n        manifest.append(dict(model=name,archive_member=member,sha256=hashlib.sha256(cached.read_bytes()).hexdigest()))\\\\n        rows.append(analyse(pd.read_csv(cached),points,out,name));print(name,\\\\\\\'complete\\\\\\\',flush=True)\\\\n    pd.DataFrame(rows).to_csv(root/\\\\\\\'six_models_summary.csv\\\\\\\',index=False)\\\\n    (root/\\\\\\\'input_manifest.json\\\\\\\').write_text(json.dumps(manifest,indent=2),encoding=\\\\\\\'utf8\\\\\\\')\\\\n    (root/\\\\\\\'method_notes.txt\\\\\\\').write_text(\\\\\\\'Reference: Yousefi & Carranza, Computers & Geosciences 79 (2015), 69–81, sections 2.3–2.4.\\\\\\\\nAdaptation: cumulative score-threshold sweep for continuous model outputs, rather than reproducing the paper\\\\\\\\\\\\\\\'s fractal discretization of evidence layers. Prediction rate counts deposit records, not expanded positive pixels. Nd=Pr/Oa. Opposing Y axes intersect where Pr+Oa=100.\\\\\\\\nCrossing thresholds are descriptive map delineation, not independently optimized or validated classification thresholds. Report full curves and missing-data coverage; do not compare conditional coverage as if identical. Pixel area assumed equal; georeferenced area weights are needed for true geographical area. No three/five-class probability scheme is imposed.\\\\\\\\n\\\\\\\',encoding=\\\\\\\'utf8\\\\\\\')\\\\nif __name__==\\\\\\\'__main__\\\\\\\':main()\\\\n\\\\n\\\'\\nPUBLICATION_HELPER_SOURCE = \\\'"""Four publication figures from manuscript-matched evaluation samples.\\\\n\\\\nFirst run PAC_manuscript.py to rebuild and audit the evaluation samples.\\\\nThis script performs no training and never changes scores or labels.\\\\n"""\\\\nfrom pathlib import Path\\\\nimport os, sys, argparse\\\\nBASE=Path(__file__).parent\\\\nos.environ.setdefault(\\\\\\\'MPLCONFIGDIR\\\\\\\',str(BASE/\\\\\\\'mpl_config\\\\\\\'))\\\\nsys.path.insert(0,str(BASE/\\\\\\\'python_dependencies\\\\\\\'))\\\\nimport numpy as np\\\\nimport pandas as pd\\\\nimport matplotlib\\\\nmatplotlib.use(\\\\\\\'Agg\\\\\\\')\\\\nimport matplotlib.pyplot as plt\\\\nfrom matplotlib.lines import Line2D\\\\nfrom matplotlib.ticker import MultipleLocator\\\\n\\\\nMODELS=[\\\\\\\'original-RF\\\\\\\',\\\\\\\'PCA-RF\\\\\\\',\\\\\\\'original-1DCNN\\\\\\\',\\\\\\\'PCA-1DCNN\\\\\\\',\\\\\\\'original-2DCNN\\\\\\\',\\\\\\\'PCA-transformed 2DCNN\\\\\\\']\\\\nLABELS=[\\\\\\\'Original RF\\\\\\\',\\\\\\\'PCA RF\\\\\\\',\\\\\\\'Original 1-D CNN\\\\\\\',\\\\\\\'PCA 1-D CNN\\\\\\\',\\\\\\\'Original 2-D CNN\\\\\\\',\\\\\\\'PCA 2-D CNN\\\\\\\']\\\\n# Model family is encoded by colour; PCA is distinguished by a dashed line.\\\\nCOLORS=[\\\\\\\'#2166AC\\\\\\\',\\\\\\\'#2166AC\\\\\\\',\\\\\\\'#238B45\\\\\\\',\\\\\\\'#238B45\\\\\\\',\\\\\\\'#B35806\\\\\\\',\\\\\\\'#B35806\\\\\\\']\\\\nplt.rcParams.update({\\\\\\\'font.family\\\\\\\':\\\\\\\'Times New Roman\\\\\\\',\\\\\\\'font.size\\\\\\\':12,\\\\n \\\\\\\'axes.titlesize\\\\\\\':17,\\\\\\\'axes.labelsize\\\\\\\':14,\\\\\\\'legend.fontsize\\\\\\\':11,\\\\n \\\\\\\'axes.linewidth\\\\\\\':.9,\\\\\\\'svg.fonttype\\\\\\\':\\\\\\\'none\\\\\\\',\\\\\\\'pdf.fonttype\\\\\\\':42,\\\\n \\\\\\\'xtick.direction\\\\\\\':\\\\\\\'out\\\\\\\',\\\\\\\'ytick.direction\\\\\\\':\\\\\\\'out\\\\\\\',\\\\\\\'savefig.facecolor\\\\\\\':\\\\\\\'white\\\\\\\'})\\\\n\\\\ndef style_axes(ax):\\\\n # Keep all four borders visible, with quiet gridlines behind the data.\\\\n for spine in ax.spines.values():\\\\n  spine.set_visible(True)\\\\n  spine.set_color(\\\\\\\'#404852\\\\\\\')\\\\n  spine.set_linewidth(.9)\\\\n ax.grid(color=\\\\\\\'#E8EBEF\\\\\\\', linewidth=.6)\\\\n ax.set_axisbelow(True)\\\\n ax.tick_params(direction=\\\\\\\'out\\\\\\\', length=4, width=.8, colors=\\\\\\\'#303740\\\\\\\', pad=5)\\\\n ax.xaxis.labelpad=9\\\\n ax.yaxis.labelpad=9\\\\n\\\\n\\\\ndef main(source,output):\\\\n output.mkdir(parents=True,exist_ok=True)\\\\n audit=pd.read_csv(source/\\\\\\\'metrics_and_ROC_audit.csv\\\\\\\').set_index(\\\\\\\'model\\\\\\\')\\\\n if not audit.ROC_recomputed.all():raise ValueError(\\\\\\\'ROC audit failed; resolve input mismatch first\\\\\\\')\\\\n figures={}\\\\n for kind in [\\\\\\\'ROC\\\\\\\',\\\\\\\'Balanced_accuracy\\\\\\\',\\\\\\\'Success_rate\\\\\\\',\\\\\\\'PR_AUC\\\\\\\']:\\\\n  fig,ax=plt.subplots(figsize=(8.2,7.3));fig.subplots_adjust(left=.13,right=.96,top=.90,bottom=.30)\\\\n  style_axes(ax)\\\\n  ax.grid(color=\\\\\\\'#E3E7EB\\\\\\\',linewidth=.65);ax.set_axisbelow(True)\\\\n  figures[kind]=(fig,ax)\\\\n rows=[]\\\\n for i,(model,label,color) in enumerate(zip(MODELS,LABELS,COLORS)):\\\\n  d=pd.read_csv(source/(model+\\\\\\\'_evaluation_samples.csv\\\\\\\'));y=d.Label.to_numpy(int);s=d.score.to_numpy(float)\\\\n  c=pd.read_csv(source/(model+\\\\\\\'_curves.csv\\\\\\\'))\\\\n  roc=float(np.trapezoid(c.recall,c.FPR))\\\\n  if abs(roc-audit.loc[model,\\\\\\\'reference_ROC_AUC\\\\\\\'])>5e-6:raise ValueError(model+\\\\\\\' ROC mismatch\\\\\\\')\\\\n  pra=float(np.trapezoid(c.precision,c.recall))\\\\n  style=dict(color=color,lw=2.0,ls=\\\\\\\'--\\\\\\\' if i%2 else \\\\\\\'-\\\\\\\',alpha=.95)\\\\n  figures[\\\\\\\'ROC\\\\\\\'][1].plot(c.FPR,c.recall,label=f\\\\\\\'{label}   AUC = {roc:.4f}\\\\\\\',**style)\\\\n  figures[\\\\\\\'PR_AUC\\\\\\\'][1].plot(c.recall,c.precision,label=f\\\\\\\'{label}   PR-AUC = {pra:.4f}\\\\\\\',**style)\\\\n  ts=np.linspace(0,1,501)\\\\n  ba=np.array([.5*(np.mean(s[y==1]>=t)+np.mean(s[y==0]<t)) for t in ts])\\\\n  ba05=.5*(np.mean(s[y==1]>=.5)+np.mean(s[y==0]<.5))\\\\n  figures[\\\\\\\'Balanced_accuracy\\\\\\\'][1].plot(ts,ba,label=f\\\\\\\'{label}   BA at 0.5 = {ba05:.4f}\\\\\\\',**style)\\\\n  figures[\\\\\\\'Success_rate\\\\\\\'][1].plot(c.evaluation_unit_fraction*100,c.recall*100,label=label,**style)\\\\n  rows.append(dict(model=model,ROC_AUC=roc,PR_AUC_trapezoidal=pra,AP=audit.loc[model,\\\\\\\'AP\\\\\\\'],balanced_accuracy_at_0_5=ba05,n=len(y),positive_prevalence=y.mean()))\\\\n  pd.DataFrame(dict(threshold=ts,balanced_accuracy=ba)).to_csv(output/(model+\\\\\\\'_BA.csv\\\\\\\'),index=False)\\\\n configs={\\\\n \\\\\\\'ROC\\\\\\\':(\\\\\\\'ROC curves (descriptive)\\\\\\\',\\\\\\\'False positive rate\\\\\\\',\\\\\\\'True positive rate\\\\\\\'),\\\\n \\\\\\\'Balanced_accuracy\\\\\\\':(\\\\\\\'Balanced accuracy\\\\\\\',\\\\\\\'Classification threshold\\\\\\\',\\\\\\\'Balanced accuracy\\\\\\\'),\\\\n \\\\\\\'Success_rate\\\\\\\':(\\\\\\\'Success-rate curves\\\\\\\',\\\\\\\'Selected evaluation units (%)\\\\\\\',\\\\\\\'Positive units captured (%)\\\\\\\'),\\\\n \\\\\\\'PR_AUC\\\\\\\':(\\\\\\\'Precision–recall curves\\\\\\\',\\\\\\\'Recall\\\\\\\',\\\\\\\'Precision\\\\\\\')}\\\\n # Symmetric four-panel layout; lift omitted because it duplicates success data.\\\\n combined=plt.figure(figsize=(13.2,10.6))\\\\n grid=combined.add_gridspec(2,2,left=.085,right=.975,top=.935,bottom=.16,wspace=.26,hspace=.32)\\\\n slots=[grid[0,0],grid[0,1],grid[1,0],grid[1,1]]\\\\n combined_order=[\\\\\\\'ROC\\\\\\\',\\\\\\\'Balanced_accuracy\\\\\\\',\\\\\\\'Success_rate\\\\\\\',\\\\\\\'PR_AUC\\\\\\\']\\\\n for kind,(fig,ax) in figures.items():\\\\n  title,xlab,ylab=configs[kind];ax.set(title=title,xlabel=xlab,ylabel=ylab)\\\\n  ax.set_xlim(0,100 if kind==\\\\\\\'Success_rate\\\\\\\' else 1)\\\\n  ax.set_ylim(0,102 if kind==\\\\\\\'Success_rate\\\\\\\' else 1.025)\\\\n  ax.xaxis.set_major_locator(MultipleLocator(20 if kind==\\\\\\\'Success_rate\\\\\\\' else .2))\\\\n  ax.yaxis.set_major_locator(MultipleLocator(20 if kind==\\\\\\\'Success_rate\\\\\\\' else .2))\\\\n  if kind==\\\\\\\'ROC\\\\\\\':ax.plot([0,1],[0,1],color=\\\\\\\'#87929C\\\\\\\',lw=1,ls=\\\\\\\':\\\\\\\',zorder=0)\\\\n  if kind==\\\\\\\'Success_rate\\\\\\\':ax.plot([0,100],[0,100],color=\\\\\\\'#87929C\\\\\\\',lw=1,ls=\\\\\\\':\\\\\\\',zorder=0)\\\\n  if kind==\\\\\\\'Balanced_accuracy\\\\\\\':\\\\n   ax.axvline(.5,color=\\\\\\\'#87929C\\\\\\\',lw=1,ls=\\\\\\\':\\\\\\\',zorder=0)\\\\n   ax.axhline(.5,color=\\\\\\\'#B7BEC5\\\\\\\',lw=.8,ls=\\\\\\\':\\\\\\\',zorder=0)\\\\n  ax.legend(loc=\\\\\\\'upper center\\\\\\\',bbox_to_anchor=(.5,-.19),ncol=2,frameon=False,handlelength=3,columnspacing=1.5,labelspacing=.8)\\\\n  for ext in [\\\\\\\'jpg\\\\\\\',\\\\\\\'svg\\\\\\\']:fig.savefig(output/(kind+\\\\\\\'.\\\\\\\'+ext),dpi=350)\\\\n  # Copy plotted data and styles, never rasterize the single-panel images.\\\\n  position=combined_order.index(kind)\\\\n  target=combined.add_subplot(slots[position])\\\\n  for line in ax.lines:\\\\n   target.plot(line.get_xdata(),line.get_ydata(),color=line.get_color(),\\\\n               linestyle=line.get_linestyle(),linewidth=1.9 if line.get_linewidth()>1 else .85,\\\\n               alpha=line.get_alpha(),label=line.get_label())\\\\n  target.set(xscale=ax.get_xscale(),yscale=ax.get_yscale(),xlim=ax.get_xlim(),ylim=ax.get_ylim(),\\\\n             xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel())\\\\n  target.set_title(title,fontsize=15,pad=13)\\\\n  target.xaxis.label.set_size(11);target.yaxis.label.set_size(11)\\\\n  target.tick_params(labelsize=10)\\\\n  style_axes(target)\\\\n  target.grid(color=\\\\\\\'#E8EBEF\\\\\\\',linewidth=.6);target.set_axisbelow(True)\\\\n  target.text(-.105,1.035,f\\\\\\\'({chr(97+position)})\\\\\\\',transform=target.transAxes,\\\\n              fontsize=15,fontweight=\\\\\\\'bold\\\\\\\',va=\\\\\\\'bottom\\\\\\\')\\\\n  if kind in [\\\\\\\'ROC\\\\\\\',\\\\\\\'PR_AUC\\\\\\\',\\\\\\\'Balanced_accuracy\\\\\\\']:\\\\n   field={\\\\\\\'ROC\\\\\\\':\\\\\\\'ROC_AUC\\\\\\\',\\\\\\\'PR_AUC\\\\\\\':\\\\\\\'PR_AUC_trapezoidal\\\\\\\',\\\\\\\'Balanced_accuracy\\\\\\\':\\\\\\\'balanced_accuracy_at_0_5\\\\\\\'}[kind]\\\\n   heading={\\\\\\\'ROC\\\\\\\':\\\\\\\'AUC\\\\\\\',\\\\\\\'PR_AUC\\\\\\\':\\\\\\\'PR-AUC\\\\\\\',\\\\\\\'Balanced_accuracy\\\\\\\':\\\\\\\'BA at 0.5\\\\\\\'}[kind]\\\\n   annotations=\\\\\\\'\\\\\\\\n\\\\\\\'.join(f\\\\\\\'{label}: {row[field]:.4f}\\\\\\\' for label,row in zip(LABELS,rows))\\\\n   target.text(.97,.06 if kind!=\\\\\\\'PR_AUC\\\\\\\' else .32,heading+\\\\\\\'\\\\\\\\n\\\\\\\'+annotations,\\\\n               transform=target.transAxes,ha=\\\\\\\'right\\\\\\\',va=\\\\\\\'bottom\\\\\\\',fontsize=9,\\\\n               linespacing=1.4,bbox=dict(boxstyle=\\\\\\\'round,pad=0.6,rounding_size=0.15\\\\\\\',facecolor=\\\\\\\'white\\\\\\\',alpha=.96,edgecolor=\\\\\\\'#D2D8DF\\\\\\\',linewidth=.8))\\\\n  plt.close(fig)\\\\n handles=[Line2D([0],[0],color=color,lw=2,ls=\\\\\\\'--\\\\\\\' if i%2 else \\\\\\\'-\\\\\\\',label=label)\\\\n          for i,(label,color) in enumerate(zip(LABELS,COLORS))]\\\\n combined.legend(handles=handles,loc=\\\\\\\'lower center\\\\\\\',bbox_to_anchor=(.5,.025),\\\\n                 ncol=3,frameon=False,fontsize=12,handlelength=3,columnspacing=2.5,labelspacing=.9)\\\\n for ext in [\\\\\\\'jpg\\\\\\\',\\\\\\\'svg\\\\\\\']:combined.savefig(output/(\\\\\\\'All_four_metrics.\\\\\\\'+ext),dpi=300)\\\\n plt.close(combined)\\\\n pd.DataFrame(rows).to_csv(output/\\\\\\\'metrics.csv\\\\\\\',index=False)\\\\n (output/\\\\\\\'README.txt\\\\\\\').write_text(\\\\\\\'Four panels: ROC, balanced accuracy, success rate, precision-recall. Lift omitted because it is derived from success rate and extreme small-sample values reduce readability. ROC-AUC verified against current-model descriptive ROC. Original/PCA share family colours with solid/dashed lines. PR-AUC is trapezoidal; AP separately reported. BA threshold=0.5. Success uses ROC evaluation units: labelled pixels for RF/1DCNN, 9x9 patches for 2DCNN. X is sample fraction, not geographic area. Different populations and prevalences limit comparison, especially PR-AUC. Legacy descriptive evaluation, not independent validation. No smoothing or changes to scores. Independent validation is NOT provided.\\\\\\\\n\\\\\\\',encoding=\\\\\\\'utf8\\\\\\\')\\\\n print(pd.DataFrame(rows).to_string(index=False))\\\\n\\\\nif __name__==\\\\\\\'__main__\\\\\\\':\\\\n p=argparse.ArgumentParser();p.add_argument(\\\\\\\'--input\\\\\\\',type=Path,default=BASE/\\\\\\\'PAC_manuscript_results\\\\\\\');p.add_argument(\\\\\\\'--output\\\\\\\',type=Path,default=BASE/\\\\\\\'PAC_four_figures\\\\\\\');a=p.parse_args();main(a.input,a.output)\\\\n\\\'\\nMAP_DRAW_SOURCE = "def draw(ax, scheme, branch, kind, letter=None, small=False):\\\\n    z = maps[scheme, branch, kind]\\\\n    mask = legacy2d if kind == \\\'2DCNN\\\' else full\\\\n    assert np.isfinite(z[mask]).all()\\\\n    cs = ax.tricontourf(triangles[\\\'patch\\\' if kind == \\\'2DCNN\\\' else \\\'pixel\\\'], z[mask], levels=LEVELS, cmap=\\\'coolwarm\\\')\\\\n    if hasattr(cs, \\\'set_rasterized\\\'):\\\\n        import warnings\\\\n        with warnings.catch_warnings():\\\\n            warnings.filterwarnings(\\\'ignore\\\', message=\\\'Rasterization of.*will be ignored\\\')\\\\n            cs.set_rasterized(True)\\\\n    else:\\\\n        for c in cs.collections:\\\\n            c.set_rasterized(True)\\\\n    ax.scatter(locations.Longitude, locations.Latitude, s=70 if small else 90, c=\\\'#fff000\\\', edgecolors=\\\'black\\\', linewidths=0.9, zorder=4)\\\\n    label = (\\\'Original\\\' if branch == \\\'Raw\\\' else \\\'PCA\\\') + \\\' \\\' + {\\\'RF\\\': \\\'RF\\\', \\\'1DCNN\\\': \\\'1-D CNN\\\', \\\'2DCNN\\\': \\\'2-D CNN\\\'}[kind]\\\\n    ax.set_title(\\\'Predicted probability map (\\\' + label + \\\')\\\', fontsize=14 if small else 15, pad=10)\\\\n    ax.set_xlim(55, 135)\\\\n    ax.set_ylim(35, 60)\\\\n    ax.set_xticks(np.arange(60, 131, 10))\\\\n    ax.set_yticks([40, 50, 60])\\\\n    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f\\\'{v:.0f}°E\\\'))\\\\n    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f\\\'{v:.0f}°N\\\'))\\\\n    ax.tick_params(direction=\\\'in\\\', length=6, labelsize=12 if small else 13)\\\\n    for s in ax.spines.values():\\\\n        s.set_linewidth(1)\\\\n    ax.set_aspect(\\\'auto\\\')\\\\n    ax.set_xlabel(\\\'\\\')\\\\n    ax.set_ylabel(\\\'\\\')\\\\n    if letter:\\\\n        ax.text(0.045, 0.88, letter, transform=ax.transAxes, fontsize=23, fontweight=\\\'bold\\\', ha=\\\'center\\\', va=\\\'center\\\', bbox=dict(boxstyle=\\\'circle,pad=.18\\\', facecolor=\\\'none\\\', edgecolor=\\\'#454545\\\', linewidth=0.5), zorder=5)\\\\n    cb = ax.figure.colorbar(cs, ax=ax, fraction=0.022, pad=0.045, ticks=np.arange(0, 1, 0.16))\\\\n    cb.ax.yaxis.set_label_position(\\\'left\\\')\\\\n    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: \\\'0\\\' if abs(v) < 1e-09 else f\\\'{v:.2f}\\\'))\\\\n    cb.ax.tick_params(labelsize=10 if small else 11)\\\\n    cb.set_label(\\\'Predicted probability\\\', fontsize=12 if small else 13, fontweight=\\\'bold\\\', labelpad=9)\\\\n    return cs\\\\n\\\\ndef save(fig, path):\\\\n    for ext in [\\\'jpg\\\', \\\'svg\\\']:\\\\n        fig.savefig(path.with_suffix(\\\'.\\\' + ext), dpi=220, bbox_inches=\\\'tight\\\', facecolor=\\\'white\\\')\\\\n    plt.close(fig)"\\nimport os\\nos.environ.setdefault(\\\'TF_CPP_MIN_LOG_LEVEL\\\',\\\'2\\\')\\nimport argparse,ast,gc,hashlib,importlib.metadata,json,shutil,subprocess,sys,types,zipfile\\nfrom pathlib import Path\\nimport numpy as np\\nimport pandas as pd\\nimport joblib\\nimport matplotlib\\nmatplotlib.use(\\\'Agg\\\')\\nimport matplotlib.pyplot as plt\\nfrom sklearn.ensemble import RandomForestClassifier\\nfrom sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,log_loss\\nROOT_DEFAULT=Path(\\\'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE\\\')\\nRAW=[\\\'176Hf-177Hf\\\',\\\'176Yb-177Hf\\\',\\\'206Pb-238U\\\',\\\'207Pb-235U\\\',\\\'207Pb_206Pb\\\',\\\'208Pb-232Th\\\',\\\'Al2O3\\\',\\\'CaO\\\',\\\'FeOT\\\',\\\'K2O\\\',\\\'LOI\\\',\\\'Lu176-Hf177\\\',\\\'MgO\\\',\\\'MnO\\\',\\\'Na2O\\\',\\\'P2O5\\\',\\\'SiO\\\',\\\'TDM1\\\',\\\'TDM2\\\',\\\'Th-U\\\',\\\'TiO2\\\',\\\'Hf(0)\\\',\\\'Hf(t)\\\']\\nRF_PARAMS={\\\'Raw\\\':dict(n_estimators=329,max_depth=16,min_samples_split=2,min_samples_leaf=1,bootstrap=True),\\\'PCA\\\':dict(n_estimators=248,max_depth=9,min_samples_split=2,min_samples_leaf=4,bootstrap=False)}\\nORDER=[(\\\'Raw\\\',\\\'RF\\\'),(\\\'PCA\\\',\\\'RF\\\'),(\\\'Raw\\\',\\\'1DCNN\\\'),(\\\'PCA\\\',\\\'1DCNN\\\'),(\\\'Raw\\\',\\\'2DCNN\\\'),(\\\'PCA\\\',\\\'2DCNN\\\')]\\nCANON={(\\\'Raw\\\',\\\'RF\\\'):\\\'original-RF\\\',(\\\'PCA\\\',\\\'RF\\\'):\\\'PCA-RF\\\',(\\\'Raw\\\',\\\'1DCNN\\\'):\\\'original-1DCNN\\\',(\\\'PCA\\\',\\\'1DCNN\\\'):\\\'PCA-1DCNN\\\',(\\\'Raw\\\',\\\'2DCNN\\\'):\\\'original-2DCNN\\\',(\\\'PCA\\\',\\\'2DCNN\\\'):\\\'PCA-transformed 2DCNN\\\'}\\nMODEL_MEMBERS={(\\\'Raw\\\',\\\'RF\\\'):\\\'Revised/fig/original-RF_trained_model.joblib\\\',(\\\'PCA\\\',\\\'RF\\\'):\\\'Revised/fig/PCA-RF_trained_model.joblib\\\',(\\\'Raw\\\',\\\'1DCNN\\\'):\\\'original_analyses/1DCNN/trained_model_origin_1DCNN.h5\\\',(\\\'PCA\\\',\\\'1DCNN\\\'):\\\'1DCNN_test/ensure/trained_model_1DCNN_test.h5\\\',(\\\'Raw\\\',\\\'2DCNN\\\'):\\\'original_analyses/2DCNN_model_original_analyse.h5\\\',(\\\'PCA\\\',\\\'2DCNN\\\'):\\\'2DCNN_test/ensure/2DCNN_model.h5\\\'}\\nCASES=[\\\'Original\\\',\\\'A\\\',\\\'B\\\']\\nREUSE_CASES={\\\'Original\\\':\\\'original_samples_seed42\\\',\\\'A\\\':\\\'A_positive_preserved\\\',\\\'B\\\':\\\'B_PUB_positive_preserved\\\'}\\n\\ndef sha(path):\\n h=hashlib.sha256()\\n with Path(path).open(\\\'rb\\\') as stream:\\n  for block in iter(lambda:stream.read(1024*1024),b\\\'\\\'):h.update(block)\\n return h.hexdigest()\\n\\ndef write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding=\\\'utf8\\\')\\n\\ndef save_pair(fig,folder,name):\\n folder.mkdir(parents=True,exist_ok=True)\\n for ext in [\\\'jpg\\\',\\\'svg\\\']:fig.savefig(folder/(name+\\\'.\\\'+ext),dpi=220,bbox_inches=\\\'tight\\\')\\n plt.close(fig)\\n\\ndef helper(source,name):\\n module=types.ModuleType(name);module.__file__=str(Path(__file__).resolve());exec(compile(source,name,\\\'exec\\\'),module.__dict__);return module\\n\\ndef extract_member(archive,member,cache):\\n target=cache/member;target.parent.mkdir(parents=True,exist_ok=True)\\n if not target.exists():\\n  partial=target.with_name(target.name+\\\'.partial\\\')\\n  with partial.open(\\\'wb\\\') as stream:subprocess.run([\\\'tar\\\',\\\'-xOf\\\',str(archive),\\\'central asian mapping/\\\'+member],stdout=stream,check=True)\\n  partial.replace(target)\\n return target\\n\\ndef cols_for(branch):return RAW if branch==\\\'Raw\\\' else [f\\\'PCA{i}\\\' for i in range(1,24)]\\n\\ndef legacy_tensors(data,cols,kind):\\n if kind!=\\\'2DCNN\\\':return data[cols].to_numpy(np.float32),data.Label.to_numpy(int)\\n arrays=[];labels=[]\\n for label in [1,0]:\\n  z=data.loc[data.Label==label,cols].to_numpy(np.float32);n=len(z)//81\\n  arrays.append(z[:n*81].reshape(n,9,9,23));labels.extend([label]*n)\\n return np.concatenate(arrays),np.array(labels)\\n\\ndef make_net(kind):\\n import tensorflow as tf\\n L=tf.keras.layers;inp=L.Input((23,1) if kind==\\\'1DCNN\\\' else (9,9,23))\\n if kind==\\\'1DCNN\\\':\\n  z=L.Conv1D(64,3,padding=\\\'same\\\',activation=\\\'relu\\\')(inp);z=L.BatchNormalization()(z);z=L.Conv1D(32,3,padding=\\\'same\\\',activation=\\\'relu\\\')(z);z=L.BatchNormalization()(z);z=L.Flatten()(z)\\n else:\\n  z=L.Conv2D(64,3,padding=\\\'same\\\',activation=\\\'relu\\\')(inp);z=L.BatchNormalization()(z);z=L.Conv2D(32,3,padding=\\\'same\\\',activation=\\\'relu\\\')(z);z=L.BatchNormalization()(z);z=L.GlobalAveragePooling2D()(z)\\n model=tf.keras.Model(inp,L.Dense(1,activation=\\\'sigmoid\\\')(z));model.compile(optimizer=\\\'adam\\\',loss=\\\'binary_crossentropy\\\',metrics=[\\\'accuracy\\\']);return model\\n\\ndef train_or_reuse(a,case,branch,kind,data,folder):\\n import tensorflow as tf\\n cols=cols_for(branch);X,y=legacy_tensors(data,cols,kind);model_file=folder/(\\\'model.joblib\\\' if kind==\\\'RF\\\' else \\\'model.keras\\\')\\n config=dict(branch=branch,model=kind,case=case,seed=0 if kind==\\\'RF\\\' else 42,positive_training_items=int(y.sum()),background_training_items=int((y==0).sum()),evaluation=\\\'resubstitution/legacy map assessment; NOT independent spatial validation\\\')\\n if a.reuse_existing:\\n  source=a.root/\\\'fig/New_negative_sample/Parameter_audit_and_controls\\\'/REUSE_CASES[case]/(branch+\\\'_\\\'+kind)\\n  source_model=source/model_file.name\\n  if not source_model.is_file():raise FileNotFoundError(\\\'Missing requested existing model: \\\'+str(source_model))\\n  if kind!=\\\'RF\\\':\\n   with zipfile.ZipFile(source_model) as z:cfg=json.loads(z.read(\\\'config.json\\\'))\\n   layers=cfg[\\\'config\\\'][\\\'layers\\\'];assert not any(\\\'Dropout\\\' in v[\\\'class_name\\\'] for v in layers)\\n   assert all(v[\\\'config\\\'].get(\\\'kernel_regularizer\\\') is None for v in layers)\\n  shutil.copy2(source_model,model_file)\\n  for name in [\\\'training_history.csv\\\',\\\'best_checkpoint.keras\\\']:\\n   if (source/name).exists():shutil.copy2(source/name,folder/name)\\n  config.update(mode=\\\'reuse_existing_verified_control\\\',source=str(source_model),source_sha256=sha(source_model))\\n else:\\n  print(\\\'TRAIN\\\',case,branch,kind,len(y),flush=True)\\n  if kind==\\\'RF\\\':\\n   m=RandomForestClassifier(**RF_PARAMS[branch],random_state=0,n_jobs=4).fit(X,y);joblib.dump(m,model_file)\\n  else:\\n   tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);m=make_net(kind);T=X.reshape(-1,23,1) if kind==\\\'1DCNN\\\' else X\\n   patience=5 if branch==\\\'Raw\\\' and kind==\\\'1DCNN\\\' else 3\\n   callbacks=[] if branch==\\\'PCA\\\' and kind==\\\'1DCNN\\\' else [tf.keras.callbacks.EarlyStopping(monitor=\\\'loss\\\',patience=patience,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor=\\\'loss\\\',factor=.1,patience=patience,min_lr=1e-6),tf.keras.callbacks.ModelCheckpoint(str(folder/\\\'best_checkpoint.keras\\\'),monitor=\\\'accuracy\\\' if kind==\\\'1DCNN\\\' else \\\'loss\\\',save_best_only=True)]\\n   h=m.fit(T,y,batch_size=32,epochs=11 if branch==\\\'PCA\\\' and kind==\\\'1DCNN\\\' else 100,callbacks=callbacks,verbose=0);pd.DataFrame(h.history).to_csv(folder/\\\'training_history.csv\\\',index=False);m.save(model_file)\\n  config[\\\'mode\\\']=\\\'new_fixed_setting_refit\\\';del m\\n if kind==\\\'RF\\\':\\n  m=joblib.load(model_file);assert all(m.get_params()[k]==v for k,v in RF_PARAMS[branch].items());config[\\\'parameters\\\']=m.get_params()\\n else:\\n  tf.keras.backend.clear_session();m=tf.keras.models.load_model(model_file,compile=False);config.update(max_epochs=11 if branch==\\\'PCA\\\' and kind==\\\'1DCNN\\\' else 100,actual_epochs=len(pd.read_csv(folder/\\\'training_history.csv\\\')),batch_size=32,optimizer=\\\'Adam default\\\',learning_rate_initial=.001,L2=0,Dropout=0)\\n config[\\\'weights_sha256\\\']=sha(model_file);write_json(folder/\\\'parameters_and_provenance.json\\\',config)\\n if kind==\\\'RF\\\':fit_score=m.predict_proba(X)[:,1]\\n else:fit_score=m.predict(X.reshape(-1,23,1) if kind==\\\'1DCNN\\\' else X,batch_size=8192 if kind==\\\'1DCNN\\\' else 256,verbose=0).ravel()\\n pd.DataFrame(dict(Label=y,score=fit_score)).to_csv(folder/\\\'training_fit_samples.csv\\\',index=False)\\n del X;return m,config\\n\\ndef infer_map(model,kind,X,patches):\\n if kind==\\\'RF\\\':return np.concatenate([model.predict_proba(X[i:i+32768])[:,1] for i in range(0,len(X),32768)]),None\\n if kind==\\\'1DCNN\\\':return model.predict(X.reshape(-1,23,1),batch_size=8192,verbose=0).ravel(),None\\n s=model.predict(patches,batch_size=256,verbose=0).ravel();image=np.full((564,1800),np.nan,np.float32);image[:558]=np.repeat(np.repeat(s.reshape(62,200),9,axis=0),9,axis=1);return image.ravel(),s\\n\\ndef tied_curves(y,s):\\n order=np.argsort(-s,kind=\\\'stable\\\');sy=s[order];yy=y[order];ends=np.r_[np.flatnonzero(np.diff(sy)!=0),len(sy)-1];tp=np.r_[0,np.cumsum(yy)[ends]];fp=np.r_[0,ends+1-np.cumsum(yy)[ends]]\\n recall=tp/y.sum();fpr=fp/(len(y)-y.sum());precision=np.divide(tp,tp+fp,out=np.ones(len(tp)),where=tp+fp>0)\\n return pd.DataFrame(dict(threshold=np.r_[np.inf,sy[ends]],recall=recall,FPR=fpr,precision=precision,evaluation_unit_fraction=(tp+fp)/len(y),lift=np.divide(recall,(tp+fp)/len(y),out=np.full(len(tp),np.nan),where=(tp+fp)>0)))\\n\\ndef metric_row(y,s):\\n h=s>=.5;return dict(n=len(y),positives=int(y.sum()),background=int((y==0).sum()),ROC_AUC=roc_auc_score(y,s),AP=average_precision_score(y,s),accuracy=accuracy_score(y,h),balanced_accuracy=balanced_accuracy_score(y,h),precision=precision_score(y,h,zero_division=0),recall=recall_score(y,h),F1=f1_score(y,h,zero_division=0),Brier=float(np.mean((s-y)**2)))\\n\\ndef sample_pa(population,positive_scores,folder,name):\\n s=np.sort(population);p=np.sort(positive_scores);cut=np.unique(np.r_[0.,s,np.nextafter(s.max(),np.inf)]);area=100*(len(s)-np.searchsorted(s,cut))/len(s);capture=100*(len(p)-np.searchsorted(p,cut))/len(p);delta=capture+area-100;idx=np.flatnonzero(delta>=0)[-1];j=min(idx+1,len(delta)-1);w=0 if delta[idx]==delta[j] else delta[idx]/(delta[idx]-delta[j]);pr=capture[idx]+w*(capture[j]-capture[idx]);oa=area[idx]+w*(area[j]-area[idx]);density=np.divide(capture,area,out=np.full(len(area),np.nan),where=area>0)\\n pd.DataFrame(dict(threshold=cut,occupied_area_percent=area,positive_unit_capture_percent=capture,normalized_density=density)).to_csv(folder/(name+\\\'_PA_data.csv\\\'),index=False)\\n fig,ax=plt.subplots(figsize=(10,6));right=ax.twinx();ax.plot(cut,capture,color=\\\'tab:blue\\\',label=\\\'Positive-unit capture\\\');right.plot(cut,area,color=\\\'tab:orange\\\',label=\\\'Occupied map area\\\');ax.set(xlim=(0,1),ylim=(0,100),xlabel=\\\'Prospectivity threshold\\\',ylabel=\\\'Positive-unit capture (%)\\\',title=name+\\\' P–A (descriptive)\\\');right.set(ylim=(100,0),ylabel=\\\'Occupied area (%) — reversed\\\');ax.text(.04,.08,f\\\'Interpolated intersection\\\\nPrediction rate = {pr:.2f}%\\\\nOccupied area = {oa:.3f}%\\\\nNormalized density = {pr/oa:.2f}\\\',transform=ax.transAxes,bbox=dict(facecolor=\\\'white\\\',edgecolor=\\\'0.7\\\'));ax.legend(loc=\\\'upper left\\\');right.legend(loc=\\\'upper right\\\');save_pair(fig,folder,name+\\\'_PA\\\')\\n return dict(intersection_prediction_rate=pr,intersection_occupied_area=oa,normalized_density=pr/oa)\\n\\ndef evaluate_map(labels,score,patch_score,kind,name,folder,audit_folder,full_frame):\\n if kind==\\\'2DCNN\\\':\\n  y=(labels.reshape(564,1800)[:558].reshape(62,9,200,9).transpose(0,2,1,3)==1).any(axis=(2,3)).ravel().astype(int);s=patch_score\\n  yy,xx=np.meshgrid(np.arange(62),np.arange(200),indexing=\\\'ij\\\');e=pd.DataFrame(dict(x=xx.ravel()*9+4,y=yy.ravel()*9+4,Label=y,score=s));unit=\\\'complete 9x9 map patch; any positive pixel; unlabeled-only patches treated as background\\\'\\n else:\\n  mask=np.isin(labels,[0,1])&np.isfinite(score);y=labels[mask].astype(int);s=score[mask];e=full_frame.loc[mask,[\\\'x\\\',\\\'y\\\']].copy();e[\\\'Label\\\']=y;e[\\\'score\\\']=s;unit=\\\'labelled training pixel (resubstitution)\\\'\\n assert len(np.unique(y))==2\\n e.to_csv(audit_folder/(name+\\\'_evaluation_samples.csv\\\'),index=False);curves=tied_curves(y,s);curves.to_csv(audit_folder/(name+\\\'_curves.csv\\\'),index=False)\\n auc_manual=float(np.trapezoid(curves.recall,curves.FPR));auc_sk=roc_auc_score(y,s);assert abs(auc_manual-auc_sk)<1e-10\\n row=metric_row(y,s);row.update(model=name,evaluation_unit=unit,reference_ROC_AUC=auc_sk,ROC_recomputed=True,BA_at_0_5=row[\\\'balanced_accuracy\\\'],PR_AUC=float(np.trapezoid(curves.precision,curves.recall)),independent_validation=False)\\n fig,ax=plt.subplots(figsize=(7,6));ax.plot(curves.FPR,curves.recall,label=f\\\'AUC = {auc_sk:.4f}\\\');ax.plot([0,1],[0,1],\\\'--\\\',color=\\\'gray\\\');ax.set(xlabel=\\\'False positive rate\\\',ylabel=\\\'True positive rate\\\',title=name+\\\' descriptive ROC\\\');ax.legend();save_pair(fig,folder,\\\'ROC\\\')\\n cm=confusion_matrix(y,s>=.5,labels=[0,1]);pd.DataFrame(cm,index=[\\\'background\\\',\\\'positive\\\'],columns=[\\\'predicted_background\\\',\\\'predicted_positive\\\']).to_csv(folder/\\\'confusion_matrix.csv\\\');fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap=\\\'Blues\\\');ax.set(xticks=[0,1],yticks=[0,1],xticklabels=[\\\'Background\\\',\\\'Positive\\\'],yticklabels=[\\\'Background\\\',\\\'Positive\\\'],xlabel=\\\'Predicted\\\',ylabel=\\\'Observed\\\',title=name+\\\' (descriptive)\\\')\\n for i in range(2):\\n  for j in range(2):ax.text(j,i,str(cm[i,j]),ha=\\\'center\\\',va=\\\'center\\\',color=\\\'white\\\' if cm[i,j]>cm.max()/2 else \\\'black\\\')\\n save_pair(fig,folder,\\\'confusion_matrix\\\')\\n row.update(sample_pa(s if kind==\\\'2DCNN\\\' else score[np.isfinite(score)],s[y==1],folder,\\\'positive_units\\\'))\\n write_json(folder/\\\'descriptive_metrics.json\\\',row);return row\\n\\ndef feature_analysis(model,kind,X,patch,baseline,folder,cols,skip=False):\\n if kind==\\\'RF\\\':values=model.feature_importances_;method=\\\'Gini importance, not causal contribution\\\'\\n elif skip:\\n  write_json(folder/\\\'sensitivity_SKIPPED.json\\\',dict(reason=\\\'Explicit --skip-sensitivity option; incomplete sensitivity analysis\\\'));return\\n else:\\n  values=[];inputs=X.reshape(-1,23,1) if kind==\\\'1DCNN\\\' else patch\\n  fingerprint=hashlib.sha256(b\\\'\\\'.join(v.numpy().tobytes() for v in model.weights)).hexdigest()\\n  for j in range(23):\\n   changed=inputs.copy()\\n   if kind==\\\'1DCNN\\\':changed[:,j,:]=0\\n   else:changed[:,:,:,j]=0\\n   s=model.predict(changed,batch_size=8192 if kind==\\\'1DCNN\\\' else 256,verbose=0).ravel();values.append(float(np.mean(np.abs(s-baseline))));del changed,s\\n   print(\\\'  zero-ablation\\\',folder.parent.name,folder.name,j+1,\\\'/23\\\',flush=True)\\n  assert fingerprint==hashlib.sha256(b\\\'\\\'.join(v.numpy().tobytes() for v in model.weights)).hexdigest();method=\\\'Full map mean absolute prediction change after setting one feature/channel to zero; not causal importance\\\'\\n table=pd.DataFrame(dict(feature=cols,importance=values));table.to_csv(folder/\\\'feature_contributions.csv\\\',index=False);write_json(folder/\\\'feature_method.json\\\',dict(method=method,population=\\\'full pixel grid\\\' if kind!=\\\'2DCNN\\\' else \\\'12400 complete 9x9 spatial map windows\\\',normalized=False))\\n fig,ax=plt.subplots(figsize=(12,5));ax.bar(np.arange(23),values,color=\\\'skyblue\\\',edgecolor=\\\'black\\\');ax.set_xticks(np.arange(23),cols,rotation=65,ha=\\\'right\\\');ax.set(ylabel=\\\'Gini importance\\\' if kind==\\\'RF\\\' else \\\'Mean absolute probability change\\\',title=folder.name+\\\' feature contribution (descriptive)\\\');fig.tight_layout();save_pair(fig,folder,\\\'feature_contributions\\\')\\n\\ndef history_figures(folder):\\n p=folder/\\\'training_history.csv\\\'\\n if not p.exists():return\\n h=pd.read_csv(p)\\n for field in [\\\'accuracy\\\',\\\'loss\\\']:\\n  fig,ax=plt.subplots(figsize=(7,4));ax.plot(np.arange(1,len(h)+1),h[field]);ax.set(xlabel=\\\'Epoch\\\',ylabel=field,title=folder.name+\\\' training \\\'+field);save_pair(fig,folder,\\\'training_\\\'+field)\\ndef render_figures(out,points,reference_enabled):\\n import matplotlib.tri as mtri\\n from matplotlib.ticker import FuncFormatter\\n cases=CASES+([\\\'Archived\\\'] if reference_enabled else []);allmaps={}\\n for case in cases:\\n  for branch,kind in ORDER:allmaps[case,branch,kind]=np.load(out/case/(branch+\\\'_\\\'+kind)/\\\'full_map_predictions.npz\\\')[\\\'score\\\']\\n y,x=np.indices((564,1800));full=np.ones(x.size,bool);legacy=((y<558)&(x%9<8)&(y%9<8)).ravel();triangles={}\\n for name,mask in [(\\\'pixel\\\',full),(\\\'patch\\\',legacy)]:triangles[name]=mtri.Triangulation(55+x.ravel()[mask]/1799*80,35+y.ravel()[mask]/563*25)\\n ns=dict(np=np,plt=plt,FuncFormatter=FuncFormatter,locations=points,LEVELS=np.arange(0,1.040001,.08),maps=allmaps,full=full,legacy2d=legacy,triangles=triangles)\\n exec(compile(MAP_DRAW_SOURCE,\\\'embedded_figure5_renderer\\\',\\\'exec\\\'),ns);draw=ns[\\\'draw\\\'];save=ns[\\\'save\\\']\\n plt.rcParams.update({\\\'font.family\\\':\\\'Arial\\\',\\\'font.weight\\\':\\\'bold\\\',\\\'axes.titleweight\\\':\\\'bold\\\',\\\'svg.fonttype\\\':\\\'none\\\'})\\n for case in cases:\\n  title=\\\'ARCHIVED WEIGHTS — historical reproduction reference\\\' if case==\\\'Archived\\\' else (\\\'ORIGINAL DATA, original settings; CNN seed42\\\' if case==\\\'Original\\\' else f\\\'CONTROL {case}: unchanged original positives; frozen new negatives; original settings; descriptive only\\\')\\n  if case==\\\'A\\\':title=\\\'SELECTED A: reviewer-enhanced negatives; Raw2D max28, PCA2D max15; descriptive only\\\'\\n  if case==\\\'B\\\':title=\\\'SELECTED B: frozen PUB negatives; original settings; seed42; descriptive only\\\'\\n  fig,axes=plt.subplots(3,2,figsize=(21.5,13.9));fig.subplots_adjust(left=.045,right=.98,bottom=.045,top=.93,wspace=.20,hspace=.42);fig.suptitle(title,fontsize=12,y=.986)\\n  for i,((branch,kind),ax) in enumerate(zip(ORDER,axes.ravel())):draw(ax,case,branch,kind,\\\'abcdef\\\'[i],small=True)\\n  save(fig,out/(\\\'Figure_5_\\\'+case))\\n  for i,(branch,kind) in enumerate(ORDER):\\n   fig,ax=plt.subplots(figsize=(10.7,4));draw(ax,case,branch,kind,\\\'abcdef\\\'[i]);save(fig,out/case/(branch+\\\'_\\\'+kind)/\\\'prediction_map\\\')\\n  print(\\\'FIGURE5\\\',case,\\\'complete\\\',flush=True)\\n # Shared-grid map stability: comparison only, not model selection.\\n from scipy.stats import spearmanr\\n xx,yy=np.meshgrid(np.arange(4,1796,9),np.arange(4,560,9));idx=yy.ravel()*1800+xx.ravel();rows=[]\\n for branch,kind in ORDER:\\n  for a,b in [(\\\'A\\\',\\\'B\\\')]+([(\\\'Original\\\',\\\'Archived\\\'),(\\\'A\\\',\\\'Archived\\\'),(\\\'B\\\',\\\'Archived\\\')] if reference_enabled else []):\\n   va=allmaps[a,branch,kind][idx];vb=allmaps[b,branch,kind][idx];n=len(va)//10;ia=set(np.argsort(va)[-n:]);ib=set(np.argsort(vb)[-n:]);rows.append(dict(branch=branch,model=kind,left=a,right=b,spearman=float(spearmanr(va,vb).statistic),MAE=float(np.mean(abs(va-vb))),top10_jaccard=len(ia&ib)/len(ia|ib),left_fraction_ge_05=float((va>=.5).mean()),right_fraction_ge_05=float((vb>=.5).mean())))\\n pd.DataFrame(rows).to_csv(out/\\\'map_stability_and_archive_comparison.csv\\\',index=False)\\n del allmaps,triangles;gc.collect()\\n\\ndef main():\\n import tensorflow as tf\\n parser=argparse.ArgumentParser(description=__doc__)\\n parser.add_argument(\\\'--root\\\',type=Path,default=ROOT_DEFAULT)\\n parser.add_argument(\\\'--archive\\\',type=Path,default=Path(\\\'D:/central asian mapping.rar\\\'))\\n parser.add_argument(\\\'--output\\\',type=Path,default=ROOT_DEFAULT/\\\'fig/New_negative_sample/Full_compatibility_reproduction\\\')\\n parser.add_argument(\\\'--reuse-existing\\\',action=\\\'store_true\\\',help=\\\'Verify and reuse the completed 18 control models; all inference and downstream analyses rerun.\\\')\\n parser.add_argument(\\\'--skip-sensitivity\\\',action=\\\'store_true\\\',help=\\\'Explicitly skip CNN full-map ablation; not a complete sensitivity run.\\\')\\n parser.add_argument(\\\'--no-archived-reference\\\',action=\\\'store_true\\\',help=\\\'Do not render old-weight reference maps.\\\')\\n a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)\\n if not a.archive.exists():raise FileNotFoundError(a.archive)\\n if not hasattr(np,\\\'trapezoid\\\'):np.trapezoid=np.trapz\\n write_json(out/\\\'RUN_STARTED.json\\\',dict(status=\\\'running\\\',reuse_existing=a.reuse_existing,all_ROC_are_descriptive=True))\\n write_json(out/\\\'COMPLETE.json\\\',dict(status=\\\'running\\\',reuse_existing=a.reuse_existing))\\n cache=out/\\\'00_source_archive\\\';points=pd.read_excel(a.root/\\\'data/data/Sample Location_delet_orogenic_gold.xlsx\\\')\\n points.to_csv(out/\\\'deposit_records.csv\\\',index=False)\\n for case in [\\\'A\\\',\\\'B\\\']:\\n  xy=np.array(FROZEN_NEGATIVE_CENTERS[case],int);assert len(xy)==101 and len(np.unique(xy,axis=0))==101\\n  d=pd.DataFrame(xy,columns=[\\\'x\\\',\\\'y\\\']);d[\\\'Longitude\\\']=55+d.x/1799*80;d[\\\'Latitude\\\']=35+d.y/563*25;d.to_csv(out/(case+\\\'_frozen_negative_centers.csv\\\'),index=False)\\n protocol=dict(version=\\\'Original-positive-label compatibility reproduction\\\',mode=\\\'existing verified control weights; new inference and all analysis\\\' if a.reuse_existing else \\\'18 original-setting refits; new inference and analysis\\\',cases=CASES,negative_sampling=\\\'Exact 101 A and 101 B centers embedded in this code; no resampling in reproduction\\\',A_design=\\\'50km exclusion around rounded deposit-grid centers; 3deg cell round robin; no geology/coverage mask\\\',B_design=\\\'Frozen previous PUB OOB low-quartile plus spatial balance; explicit adaptation, not confirmed barren samples\\\',original_positive_rows=7686,A_B_negative_rows=8181,legacy_2D=\\\'94 positive tensors and 101 A/B background tensors, original same-label groups of81; not real geographic training neighborhoods; 72 remaining positive rows discarded only by2D\\\',PCA=\\\'Reuse original 23 frozen PCA feature columns; do not refit PCA\\\',mapping=\\\'RF/1D each grid pixel;2D true spatial9x9 map windows from0,0 stride9,12400; complete9x9 numeric fill; northern rows y=558..563 NaN\\\',display=\\\'Original8x8 support then tricontourf; corrected9x9 numeric predictions retained\\\',ROC=\\\'RF/1D: labelled training pixels.2D: full12400 map windows, any known positive pixel makes positive; all remaining windows treated as background. Different populations; descriptive only\\\',PA=\\\'Both known-deposit-record capture vs full-map occupied area and positive evaluation-unit capture are provided; equal-pixel areas as latest reference\\\',RF_final_parameters=RF_PARAMS,CNN=\\\'Original conv64/32,3kernel,BN,GAP/Flatten,Adam,BCE,batch32,seed42,max100/early loss3(twoD)/loss5(Raw1D); PCA1D11 epochs no callbacks; noL2,noDropout\\\',randomness=\\\'Historical CNN seed/environment not recorded. Current fixed seed42 reproduces control condition; bitwise equality across future environments is not guaranteed\\\',independent_spatial_validation=False,standard_deviation_across_folds=\\\'Not available for this non-CV experiment; not fabricated or borrowed\\\')\\n write_json(out/\\\'protocol.json\\\',protocol)\\n versions={k:importlib.metadata.version(k) for k in [\\\'numpy\\\',\\\'pandas\\\',\\\'scikit-learn\\\',\\\'tensorflow\\\',\\\'keras\\\',\\\'matplotlib\\\',\\\'joblib\\\']};write_json(out/\\\'environment_versions.json\\\',versions)\\n pa=helper(PA_HELPER_SOURCE,\\\'embedded_current_PA\\\');publication=helper(PUBLICATION_HELPER_SOURCE,\\\'embedded_current_publication\\\')\\n metrics={c:[] for c in CASES};pa_rows={c:[] for c in CASES};training_rows=[];inputs=[];parity=[]\\n for branch,relative in [(\\\'Raw\\\',\\\'original_analyses/grayscale_data_combined_train.csv\\\'),(\\\'PCA\\\',\\\'data/PCA_with_labels_Train.csv\\\')]:\\n  old_path=extract_member(a.archive,relative,cache);old=pd.read_csv(old_path);cols=cols_for(branch)\\n  assert old.columns[-23:].tolist()==cols;assert int((old.Label==1).sum())==7686;assert not old.duplicated([\\\'x\\\',\\\'y\\\']).any();assert old.Label.isin([0,1]).all()\\n  testrel=relative.replace(\\\'_train\\\',\\\'_test\\\').replace(\\\'_Train\\\',\\\'_Test\\\');source_full=a.root/\\\'data\\\'/testrel;full=pd.read_csv(source_full).sort_values([\\\'y\\\',\\\'x\\\']).reset_index(drop=True)\\n  expected=np.column_stack([np.tile(np.arange(1800),564),np.repeat(np.arange(564),1800)])\\n  assert np.array_equal(full[[\\\'x\\\',\\\'y\\\']].values,expected);X=full[cols].to_numpy(np.float32);assert np.isfinite(X).all();map_patches=X.reshape(564,1800,23)[:558].reshape(62,9,200,9,23).transpose(0,2,1,3,4).reshape(-1,9,9,23)\\n  positive=old[old.Label==1].copy();datasets={\\\'Original\\\':old}\\n  for case in [\\\'A\\\',\\\'B\\\']:\\n   mask=np.zeros((564,1800),bool)\\n   for x,y in FROZEN_NEGATIVE_CENTERS[case]:mask[y-4:y+5,x-4:x+5]=True\\n   neg=full.loc[mask.ravel(),old.columns].copy();neg[\\\'Label\\\']=0\\n   assert len(neg)==8181\\n   assert not set(map(tuple,neg[[\\\'x\\\',\\\'y\\\']].values)) & set(map(tuple,positive[[\\\'x\\\',\\\'y\\\']].values))\\n   data=pd.concat([neg,positive],ignore_index=True);assert np.array_equal(data.loc[data.Label==1,cols].values,positive[cols].values);datasets[case]=data\\n  for case,data in datasets.items():\\n   caseout=out/case;caseout.mkdir(exist_ok=True);data.to_csv(caseout/(branch+\\\'_training_input.csv\\\'),index=False)\\n   labels=np.full(len(full),np.nan);labels[data.y.to_numpy(int)*1800+data.x.to_numpy(int)]=data.Label.values\\n   if a.reuse_existing:\\n    src=old_path if case==\\\'Original\\\' else a.root/\\\'fig/New_negative_sample/Parameter_audit_and_controls\\\'/(branch+\\\'_\\\'+(\\\'B_PUB\\\' if case==\\\'B\\\' else \\\'A\\\')+\\\'_original_positive_new_negative.csv\\\')\\n    reference=pd.read_csv(src);assert np.array_equal(reference[[\\\'x\\\',\\\'y\\\',\\\'Label\\\']].values,data[[\\\'x\\\',\\\'y\\\',\\\'Label\\\']].values)\\n    assert np.array_equal(reference[cols].to_numpy(np.float32),data[cols].to_numpy(np.float32))\\n   inputs.append(dict(case=case,branch=branch,original_train_member=relative,original_train_sha256=sha(old_path),source_features=str(source_full),positive_rows=7686,negative_rows=int((data.Label==0).sum())))\\n   auditdir=caseout/\\\'ROC_audit\\\';auditdir.mkdir(exist_ok=True)\\n   for kind in [\\\'RF\\\',\\\'1DCNN\\\',\\\'2DCNN\\\']:\\n    folder=caseout/(branch+\\\'_\\\'+kind);folder.mkdir(exist_ok=True);print(\\\'MODEL\\\',case,branch,kind,flush=True)\\n    model,config=train_or_reuse(a,case,branch,kind,data,folder);score,patch_score=infer_map(model,kind,X,map_patches)\\n    np.savez_compressed(folder/\\\'full_map_predictions.npz\\\',score=score.astype(np.float32),shape=[564,1800],extent=[55,135,35,60])\\n    frame=full[[\\\'x\\\',\\\'y\\\']].copy();frame[\\\'Label\\\']=labels;frame[\\\'Predicted_Probability\\\']=score;frame.to_csv(folder/\\\'full_predictions.csv\\\',index=False)\\n    # Verify against independently stored 12400-point controls when reusing weights.\\n    if a.reuse_existing:\\n     ref=a.root/\\\'fig/New_negative_sample/Parameter_audit_and_controls\\\'/REUSE_CASES[case]/(branch+\\\'_\\\'+kind)/\\\'common_grid_scores.csv\\\';ref=pd.read_csv(ref);ix=ref.y.to_numpy(int)*1800+ref.x.to_numpy(int);difference=float(np.max(np.abs(score[ix]-ref.score.values)));assert difference<1e-5;parity.append(dict(case=case,model=branch+\\\'_\\\'+kind,max_common_grid_difference=difference))\\n    name=CANON[branch,kind];row=evaluate_map(labels,score,patch_score,kind,name,folder,auditdir,full);row.update(case=case,branch=branch);metrics[case].append(row)\\n    paout=folder/\\\'PA_by_deposit_records\\\';paout.mkdir(exist_ok=True);pa_rows[case].append(pa.analyse(frame,points,paout,branch+\\\'_\\\'+kind))\\n    training=pd.read_csv(folder/\\\'training_fit_samples.csv\\\');tr=metric_row(training.Label.to_numpy(int),training.score.to_numpy());tr.update(case=case,branch=branch,model=kind,evaluation=\\\'training fit; never independent validation\\\');training_rows.append(tr);history_figures(folder)\\n    feature_analysis(model,kind,X,map_patches,score if kind!=\\\'2DCNN\\\' else patch_score,folder,cols,a.skip_sensitivity)\\n    pd.DataFrame(metrics[case]).to_csv(auditdir/\\\'metrics_and_ROC_audit.csv\\\',index=False);pd.DataFrame(pa_rows[case]).to_csv(caseout/\\\'PA_deposit_summary.csv\\\',index=False);pd.DataFrame(training_rows).to_csv(out/\\\'training_fit_metrics.csv\\\',index=False)\\n    write_json(out/\\\'progress.json\\\',dict(branch=branch,case=case,model=kind,completed_models=sum(map(len,metrics.values())),sensitivity_skipped=a.skip_sensitivity))\\n    del model,score,patch_score,frame;tf.keras.backend.clear_session();gc.collect()\\n  if not a.no_archived_reference:\\n   for kind in [\\\'RF\\\',\\\'1DCNN\\\',\\\'2DCNN\\\']:\\n    folder=out/\\\'Archived\\\'/(branch+\\\'_\\\'+kind);folder.mkdir(parents=True,exist_ok=True);path=extract_member(a.archive,MODEL_MEMBERS[branch,kind],cache)\\n    model=joblib.load(path) if kind==\\\'RF\\\' else tf.keras.models.load_model(path,compile=False);s,_=infer_map(model,kind,X,map_patches);np.savez_compressed(folder/\\\'full_map_predictions.npz\\\',score=s.astype(np.float32),shape=[564,1800]);write_json(folder/\\\'reference_provenance.json\\\',dict(archive_member=MODEL_MEMBERS[branch,kind],sha256=sha(path),reference_only=True,new_negative_sampling_result=False));del model,s;tf.keras.backend.clear_session()\\n  del full,X,map_patches,datasets,old,positive;gc.collect()\\n plt.rcParams.update({\\\'font.family\\\':\\\'Times New Roman\\\',\\\'font.weight\\\':\\\'normal\\\',\\\'axes.titleweight\\\':\\\'normal\\\'})\\n for case in CASES:\\n  publication.main(out/case/\\\'ROC_audit\\\',out/case/\\\'four_metrics\\\')\\n  (out/case/\\\'ROC_audit/README_descriptive.txt\\\').write_text(\\\'Current-model predictions only. ROC reference values are independently recomputed from THESE evaluation samples, not copied from the manuscript. RF/1D use labelled training pixels;2D uses12400 spatial map windows with any known positive pixel. Unlabelled-only windows treated as background. This is legacy descriptive evaluation, not independent test/CV. PR prevalences and evaluation units differ. P-A records count deposit records; ROC success counts evaluation units. Do not borrow spatial metrics from another pipeline.\\\',encoding=\\\'utf8\\\')\\n pd.concat([pd.DataFrame(metrics[c]) for c in CASES],ignore_index=True).to_csv(out/\\\'all_18_model_descriptive_metrics.csv\\\',index=False);pd.DataFrame(inputs).to_csv(out/\\\'input_audit.csv\\\',index=False);write_json(out/\\\'prediction_parity_checks.json\\\',parity)\\n render_figures(out,points,not a.no_archived_reference)\\n # Sampling overview and source-to-output stage coverage.\\n fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True,sharey=True)\\n for case,ax in zip([\\\'A\\\',\\\'B\\\'],axes):\\n  z=pd.read_csv(out/(case+\\\'_frozen_negative_centers.csv\\\'));ax.scatter(z.Longitude,z.Latitude,s=18,c=\\\'royalblue\\\',label=\\\'Frozen background centers\\\');ax.scatter(points.Longitude,points.Latitude,s=24,c=\\\'gold\\\',edgecolors=\\\'black\\\',label=\\\'Deposit records\\\');ax.set(title=\\\'Scheme \\\'+case,xlim=(55,135),ylim=(35,60),ylabel=\\\'Latitude\\\');ax.legend()\\n axes[-1].set_xlabel(\\\'Longitude\\\');fig.tight_layout();save_pair(fig,out,\\\'A_B_negative_sampling\\\')\\n coverage=[(\\\'Original labels and frozen A/B centers\\\',\\\'Reproduced; no positive reconstruction or resampling\\\'),(\\\'Six model combinations × three training cases\\\',\\\'18 refits or explicitly verified reuse\\\'),(\\\'Archived weights\\\',\\\'Reference maps only\\\'),(\\\'Latest known-deposit P–A and success-rate\\\',\\\'Embedded latest PA analyse function, current full predictions\\\'),(\\\'ROC/PR/AP/BA/success four-panel publication figures\\\',\\\'Embedded latest publication function; current score self-audit replaces manuscript AUC gate\\\'),(\\\'Positive evaluation-unit P–A\\\',\\\'Latest audit definition; current full-map population\\\'),(\\\'Feature contributions\\\',\\\'RF Gini; CNN full-map zero-channel ablation, unless explicitly skipped\\\'),(\\\'Training curves, confusion matrices, model parameters\\\',\\\'Included\\\'),(\\\'PCA re-fitting/new geological preprocessing\\\',\\\'Intentionally excluded: this version freezes original feature tables\\\'),(\\\'Independent spatial CV\\\',\\\'Not part of this compatibility experiment; never claimed\\\')]\\n pd.DataFrame(coverage,columns=[\\\'stage\\\',\\\'implementation\\\']).to_csv(out/\\\'stage_coverage.csv\\\',index=False)\\n notes=\\\'\\\'\\\'# 完整兼容版复现与后续分析\\n\\n本程序固定此前A/B各101个负中心，保留原始7686个正像元、原始输入顺序和模型设置。原始样本、A、B三组各六个模型，共18个拟合；旧权重另作制图基准。默认从头训练。--reuse-existing会核对训练输入并复用之前已完成的控制权重，但所有全区预测、ROC、P–A、敏感性及图片重新计算。--skip-sensitivity会明确标记CNN敏感性不完整。\\n\\n输出包含四套Figure5样式六面板图、18个模型的完整预测CSV/NPZ、权重与参数、训练曲线、混淆矩阵、单模型ROC、ROC/BA/样本成功率/PR四指标单图及组合图、AP与梯形PR-AUC、矿点记录P–A/成功率、评价正样本单位P–A、RF Gini、CNN全区置零敏感性、输入/参数/预测核验。JPG/SVG同时输出。原先的结果文件夹没有覆盖。\\n\\n后续图形和P–A计算取自用户提供的最新版脚本，已嵌入本文件；不依赖其他Python脚本。修改点是从当前模型预测生成评价样本，删除旧论文AUC必须相等的门槛，改为与当前样本上独立计算的AUC核对。绝不从旧存档预测提取概率冒充新A/B结果。原RAR只提供原始训练输入和标明来源的旧权重基准。\\n\\nRF/1D的ROC是训练标记像元的回代评价；2D的ROC是完整12400个真实制图窗口，只要有正像元则标正，其他窗口当背景。训练2D仍为原始同标签每81行拼块，94正块，A/B101负块；这不是空间独立验证，不能解释成独立泛化AUC。2D的72个正标签余数按原代码丢弃，RF/1D保留全部。训练拟合指标与地图评价指标分别保存。\\n\\n矿点P–A按矿点记录计捕获率，ROC成功率按评价单位计捕获率，二者横轴/分母不能混同。P–A沿用最新版等像元面积假设，无地质/覆盖掩膜；缺失预测明确统计。插值交点是描述性阈值，不是独立优化分类阈值。\\n\\n数值2D地图按完整9x9回填，北侧6行（y=558..563）无完整窗口保留NaN；为匹配原图视觉，Figure5仍采用原始8x8有效支撑和tricontourf，未把漏填缺陷重新写入数值CSV。图层概率没有拉伸或混入旧值。SVG中密集填色层嵌入图像，文字、轴和矿点为矢量。\\n\\nPCA复用原全区23维特征，不重拟合。原CNN种子未记录，本控制固定42；相同超参数不保证复原历史权重。没有跨折评价，所以没有伪造折间标准差或借用另一版的空间验证指标。A/B的指标不能当成重复试验标准差。\\n\\n结果目录Original/A/B含各自六模型和four_metrics；Archived只含旧权重参考。full_map_predictions.npz中score可reshape(564,1800)。当前程序默认完整训练和完整敏感性。复现条件、环境、输入哈希及stage_coverage均随结果导出。\\n\\\'\\\'\\\'\\n (out/\\\'README_完整复现说明.md\\\').write_text(notes,encoding=\\\'utf8\\\')\\n try:shutil.copy2(Path(__file__).resolve(),out/\\\'reproduction_code_snapshot.py\\\')\\n except shutil.SameFileError:pass\\n write_json(out/\\\'COMPLETE.json\\\',dict(status=\\\'complete\\\',fitted_or_reused_models=18,archived_reference=not a.no_archived_reference,CNN_sensitivity_complete=not a.skip_sensitivity,independent_validation=False,mode=\\\'reuse_existing\\\' if a.reuse_existing else \\\'refit\\\',script_sha256=sha(Path(__file__))))\\n print(\\\'COMPLETE:\\\',out,flush=True)\\n\\n\'\ne=types.ModuleType("analysis_engine");e.__file__=__file__;exec(compile(ENGINE_SOURCE,"analysis_engine","exec"),e.__dict__)\nPA_SOURCE=\'"""Reanalyse saved predictions only. No model loading, fitting or PCA fitting."""\\nimport argparse,json,subprocess,hashlib\\nfrom pathlib import Path\\nimport numpy as np\\nimport pandas as pd\\nimport matplotlib\\nmatplotlib.use(\\\'Agg\\\')\\nimport matplotlib.pyplot as plt\\nplt.rcParams.update({\\\'font.family\\\':\\\'Times New Roman\\\',\\\'svg.fonttype\\\':\\\'none\\\'})\\nARCHIVE=\\\'C:/central asian mapping.rar\\\'\\nFILES={\\n\\\'Original_RF\\\':\\\'Revised/fig/original-RF_predictions.csv\\\',\\n\\\'PCA_RF\\\':\\\'Revised/fig/PCA-RF_predictions.csv\\\',\\n\\\'Original_1DCNN\\\':\\\'original_analyses/1DCNN/PCA_with_predictions_origin_1DCNN.csv\\\',\\n\\\'PCA_1DCNN\\\':\\\'Revised/fig/PCA-1DCNN_predictions.csv\\\',\\n\\\'Original_2DCNN\\\':\\\'Revised/fig/original-2DCNN_predictions.csv\\\',\\n\\\'PCA_2DCNN\\\':\\\'2DCNN_test/ensure/PCA_with_full_predictions_Test_2DCNN_test.csv\\\'}\\nROOT=Path(\\\'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/data\\\')\\n\\ndef save(fig,out,name):\\n    for ext in [\\\'jpg\\\',\\\'svg\\\']:fig.savefig(out/(name+\\\'.\\\'+ext),dpi=300,bbox_inches=\\\'tight\\\')\\n    plt.close(fig)\\n\\ndef analyse(frame,points,out,name):\\n    frame=frame.rename(columns={\\\'X\\\':\\\'x\\\',\\\'Y\\\':\\\'y\\\'})\\n    candidates=[c for c in frame if \\\'probab\\\' in c.lower() or c==\\\'Predicted_Probability\\\']\\n    if len(candidates)!=1:raise ValueError(f\\\'{name}: ambiguous probability columns {candidates}\\\')\\n    col=candidates[0]\\n    frame[col]=pd.to_numeric(frame[col],errors=\\\'coerce\\\')\\n    if frame.duplicated([\\\'x\\\',\\\'y\\\']).any():raise ValueError(\\\'Duplicate pixel coordinates\\\')\\n    valid=np.isfinite(frame[col]);missing=int((~valid).sum());data=frame.loc[valid,[\\\'x\\\',\\\'y\\\',col]].copy()\\n    scores=data[col].to_numpy(); ss=np.sort(scores)\\n    if len(ss)==0:raise ValueError(\\\'No finite predictions\\\')\\n    if ss.min()<0 or ss.max()>1:raise ValueError(\\\'Scores outside probability interval\\\')\\n    lookup=data.set_index([\\\'x\\\',\\\'y\\\'])[col]\\n    deposits=points.copy();deposits[\\\'x\\\']=((deposits.Longitude-55)/80*1799).astype(int);deposits[\\\'y\\\']=((deposits.Latitude-35)/25*563).astype(int)\\n    deposits[\\\'probability\\\']=[lookup.get((x,y),np.nan) for x,y in zip(deposits.x,deposits.y)]\\n    deposits.to_csv(out/\\\'deposit_scores.csv\\\',index=False)\\n    ds=np.sort(deposits.probability.dropna().to_numpy())\\n    if not len(ds):raise ValueError(\\\'No deposits have predictions\\\')\\n    # All observed score cutoffs, with ties kept together; no arbitrary 0.1 probability bins.\\n    thresholds=np.unique(np.r_[0.,scores,np.nextafter(scores.max(),np.inf)])\\n    area=(len(ss)-np.searchsorted(ss,thresholds,side=\\\'left\\\'))/len(ss)*100\\n    capture=(len(ds)-np.searchsorted(ds,thresholds,side=\\\'left\\\'))/len(ds)*100\\n    density=np.divide(capture,area,out=np.full(len(area),np.nan),where=area>0)\\n    curve=pd.DataFrame({\\\'threshold\\\':thresholds,\\\'occupied_area_percent\\\':area,\\\'deposit_capture_percent\\\':capture,\\\'normalized_density\\\':density})\\n    curve.to_csv(out/\\\'PA_all_thresholds.csv\\\',index=False)\\n    # Reversed right axis: visual intersection satisfies Pr = 100 - Oa.\\n    delta=capture+area-100\\n    idx=np.flatnonzero(delta>=0)[-1];j=min(idx+1,len(delta)-1)\\n    weight=0 if delta[idx]==delta[j] else delta[idx]/(delta[idx]-delta[j])\\n    cross_t=thresholds[idx]+weight*(thresholds[j]-thresholds[idx])\\n    pr=capture[idx]+weight*(capture[j]-capture[idx]);oa=area[idx]+weight*(area[j]-area[idx])\\n    actual_a=float((scores>=cross_t).mean()*100);actual_p=float((ds>=cross_t).mean()*100)\\n    row=dict(model=name,intersection_threshold_interpolated=cross_t,prediction_rate_interpolated=pr,occupied_area_interpolated=oa,normalized_density_interpolated=pr/oa if oa else None,actual_threshold_area_percent=actual_a,actual_threshold_deposit_capture_percent=actual_p,actual_normalized_density=actual_p/actual_a if actual_a else None,total_map_rows=len(frame),finite_map_pixels=len(scores),missing_prediction_pixels=missing,total_deposits=len(deposits),deposits_with_prediction=len(ds),deposits_missing_prediction=len(deposits)-len(ds))\\n    fig,ax=plt.subplots(figsize=(10,6));right=ax.twinx()\\n    ax.plot(thresholds,capture,color=\\\'tab:blue\\\',label=\\\'Deposit capture\\\');right.plot(thresholds,area,color=\\\'tab:orange\\\',label=\\\'Occupied area\\\')\\n    ax.set_ylim(0,100);right.set_ylim(100,0);ax.set_xlim(0,1)\\n    ax.set_xlabel(\\\'Prospectivity threshold (score >= threshold)\\\');ax.set_ylabel(\\\'Prediction / success rate (%)\\\');right.set_ylabel(\\\'Occupied area (%) — reversed\\\')\\n    ax.axvline(cross_t,color=\\\'gray\\\',ls=\\\'--\\\');ax.set_title(name+\\\' P–A (known-deposit success assessment)\\\')\\n    ax.legend(loc=\\\'upper left\\\');right.legend(loc=\\\'upper right\\\');save(fig,out,\\\'PA\\\')\\n    fig,ax=plt.subplots(figsize=(7,6));ax.plot(area[::-1],capture[::-1],color=\\\'tab:blue\\\');ax.plot([0,100],[0,100],\\\'--\\\',color=\\\'gray\\\');ax.set(xlabel=\\\'Occupied area (%)\\\',ylabel=\\\'Known deposits captured (%)\\\',title=name+\\\' success-rate curve\\\',xlim=(0,100),ylim=(0,100));save(fig,out,\\\'success_rate\\\')\\n    (out/\\\'metrics.txt\\\').write_text(\\\'\\\\n\\\'.join(f\\\'{k} = {v}\\\' for k,v in row.items())+\\\'\\\\nDescriptive assessment using known deposits, not independent validation.\\\\nArea fractions assume equal-area pixels. Missing predictions are excluded, not filled.\\\\nThe interpolated crossing may not be exactly attainable with discrete/tied scores.\\\\n\\\',encoding=\\\'utf-8-sig\\\')\\n    return row\\n\\ndef main():\\n    p=argparse.ArgumentParser();p.add_argument(\\\'--output\\\',default=\\\'D:/homework/夏季学期/地化大数据/central asian mapping/Revised JGE/fig/P-C\\\');p.add_argument(\\\'--archive\\\',default=ARCHIVE);a=p.parse_args()\\n    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)\\n    points=pd.read_excel(ROOT/\\\'data/Sample Location_delet_orogenic_gold.xlsx\\\')\\n    rows=[];manifest=[]\\n    for name,member in FILES.items():\\n        out=root/name;out.mkdir(exist_ok=True);cached=out/\\\'saved_predictions.csv\\\'\\n        with cached.open(\\\'wb\\\') as f:\\n            subprocess.run([\\\'tar\\\',\\\'-xOf\\\',a.archive,\\\'central asian mapping/\\\'+member],stdout=f,check=True)\\n        manifest.append(dict(model=name,archive_member=member,sha256=hashlib.sha256(cached.read_bytes()).hexdigest()))\\n        rows.append(analyse(pd.read_csv(cached),points,out,name));print(name,\\\'complete\\\',flush=True)\\n    pd.DataFrame(rows).to_csv(root/\\\'six_models_summary.csv\\\',index=False)\\n    (root/\\\'input_manifest.json\\\').write_text(json.dumps(manifest,indent=2),encoding=\\\'utf8\\\')\\n    (root/\\\'method_notes.txt\\\').write_text(\\\'Reference: Yousefi & Carranza, Computers & Geosciences 79 (2015), 69–81, sections 2.3–2.4.\\\\nAdaptation: cumulative score-threshold sweep for continuous model outputs, rather than reproducing the paper\\\\\\\'s fractal discretization of evidence layers. Prediction rate counts deposit records, not expanded positive pixels. Nd=Pr/Oa. Opposing Y axes intersect where Pr+Oa=100.\\\\nCrossing thresholds are descriptive map delineation, not independently optimized or validated classification thresholds. Report full curves and missing-data coverage; do not compare conditional coverage as if identical. Pixel area assumed equal; georeferenced area weights are needed for true geographical area. No three/five-class probability scheme is imposed.\\\\n\\\',encoding=\\\'utf8\\\')\\nif __name__==\\\'__main__\\\':main()\\n\\n\'\nPUBLICATION_SOURCE=\'"""Four publication figures from manuscript-matched evaluation samples.\\n\\nFirst run PAC_manuscript.py to rebuild and audit the evaluation samples.\\nThis script performs no training and never changes scores or labels.\\n"""\\nfrom pathlib import Path\\nimport os, sys, argparse\\nBASE=Path(__file__).parent\\nos.environ.setdefault(\\\'MPLCONFIGDIR\\\',str(BASE/\\\'mpl_config\\\'))\\nsys.path.insert(0,str(BASE/\\\'python_dependencies\\\'))\\nimport numpy as np\\nimport pandas as pd\\nimport matplotlib\\nmatplotlib.use(\\\'Agg\\\')\\nimport matplotlib.pyplot as plt\\nfrom matplotlib.lines import Line2D\\nfrom matplotlib.ticker import MultipleLocator\\n\\nMODELS=[\\\'original-RF\\\',\\\'PCA-RF\\\',\\\'original-1DCNN\\\',\\\'PCA-1DCNN\\\',\\\'original-2DCNN\\\',\\\'PCA-transformed 2DCNN\\\']\\nLABELS=[\\\'Original RF\\\',\\\'PCA RF\\\',\\\'Original 1-D CNN\\\',\\\'PCA 1-D CNN\\\',\\\'Original 2-D CNN\\\',\\\'PCA 2-D CNN\\\']\\n# Model family is encoded by colour; PCA is distinguished by a dashed line.\\nCOLORS=[\\\'#2166AC\\\',\\\'#2166AC\\\',\\\'#238B45\\\',\\\'#238B45\\\',\\\'#B35806\\\',\\\'#B35806\\\']\\nplt.rcParams.update({\\\'font.family\\\':\\\'Times New Roman\\\',\\\'font.size\\\':12,\\n \\\'axes.titlesize\\\':17,\\\'axes.labelsize\\\':14,\\\'legend.fontsize\\\':11,\\n \\\'axes.linewidth\\\':.9,\\\'svg.fonttype\\\':\\\'none\\\',\\\'pdf.fonttype\\\':42,\\n \\\'xtick.direction\\\':\\\'out\\\',\\\'ytick.direction\\\':\\\'out\\\',\\\'savefig.facecolor\\\':\\\'white\\\'})\\n\\ndef style_axes(ax):\\n # Keep all four borders visible, with quiet gridlines behind the data.\\n for spine in ax.spines.values():\\n  spine.set_visible(True)\\n  spine.set_color(\\\'#404852\\\')\\n  spine.set_linewidth(.9)\\n ax.grid(color=\\\'#E8EBEF\\\', linewidth=.6)\\n ax.set_axisbelow(True)\\n ax.tick_params(direction=\\\'out\\\', length=4, width=.8, colors=\\\'#303740\\\', pad=5)\\n ax.xaxis.labelpad=9\\n ax.yaxis.labelpad=9\\n\\n\\ndef main(source,output):\\n output.mkdir(parents=True,exist_ok=True)\\n audit=pd.read_csv(source/\\\'metrics_and_ROC_audit.csv\\\').set_index(\\\'model\\\')\\n if not audit.ROC_recomputed.all():raise ValueError(\\\'ROC audit failed; resolve input mismatch first\\\')\\n figures={}\\n for kind in [\\\'ROC\\\',\\\'Balanced_accuracy\\\',\\\'Success_rate\\\',\\\'PR_AUC\\\']:\\n  fig,ax=plt.subplots(figsize=(8.2,7.3));fig.subplots_adjust(left=.13,right=.96,top=.90,bottom=.30)\\n  style_axes(ax)\\n  ax.grid(color=\\\'#E3E7EB\\\',linewidth=.65);ax.set_axisbelow(True)\\n  figures[kind]=(fig,ax)\\n rows=[]\\n for i,(model,label,color) in enumerate(zip(MODELS,LABELS,COLORS)):\\n  d=pd.read_csv(source/(model+\\\'_evaluation_samples.csv\\\'));y=d.Label.to_numpy(int);s=d.score.to_numpy(float)\\n  c=pd.read_csv(source/(model+\\\'_curves.csv\\\'))\\n  roc=float(np.trapezoid(c.recall,c.FPR))\\n  if abs(roc-audit.loc[model,\\\'reference_ROC_AUC\\\'])>5e-6:raise ValueError(model+\\\' ROC mismatch\\\')\\n  pra=float(np.trapezoid(c.precision,c.recall))\\n  style=dict(color=color,lw=2.0,ls=\\\'--\\\' if i%2 else \\\'-\\\',alpha=.95)\\n  figures[\\\'ROC\\\'][1].plot(c.FPR,c.recall,label=f\\\'{label}   AUC = {roc:.4f}\\\',**style)\\n  figures[\\\'PR_AUC\\\'][1].plot(c.recall,c.precision,label=f\\\'{label}   PR-AUC = {pra:.4f}\\\',**style)\\n  ts=np.linspace(0,1,501)\\n  ba=np.array([.5*(np.mean(s[y==1]>=t)+np.mean(s[y==0]<t)) for t in ts])\\n  ba05=.5*(np.mean(s[y==1]>=.5)+np.mean(s[y==0]<.5))\\n  figures[\\\'Balanced_accuracy\\\'][1].plot(ts,ba,label=f\\\'{label}   BA at 0.5 = {ba05:.4f}\\\',**style)\\n  figures[\\\'Success_rate\\\'][1].plot(c.evaluation_unit_fraction*100,c.recall*100,label=label,**style)\\n  rows.append(dict(model=model,ROC_AUC=roc,PR_AUC_trapezoidal=pra,AP=audit.loc[model,\\\'AP\\\'],balanced_accuracy_at_0_5=ba05,n=len(y),positive_prevalence=y.mean()))\\n  pd.DataFrame(dict(threshold=ts,balanced_accuracy=ba)).to_csv(output/(model+\\\'_BA.csv\\\'),index=False)\\n configs={\\n \\\'ROC\\\':(\\\'ROC curves (descriptive)\\\',\\\'False positive rate\\\',\\\'True positive rate\\\'),\\n \\\'Balanced_accuracy\\\':(\\\'Balanced accuracy\\\',\\\'Classification threshold\\\',\\\'Balanced accuracy\\\'),\\n \\\'Success_rate\\\':(\\\'Success-rate curves\\\',\\\'Selected evaluation units (%)\\\',\\\'Positive units captured (%)\\\'),\\n \\\'PR_AUC\\\':(\\\'Precision–recall curves\\\',\\\'Recall\\\',\\\'Precision\\\')}\\n # Symmetric four-panel layout; lift omitted because it duplicates success data.\\n combined=plt.figure(figsize=(13.2,10.6))\\n grid=combined.add_gridspec(2,2,left=.085,right=.975,top=.935,bottom=.16,wspace=.26,hspace=.32)\\n slots=[grid[0,0],grid[0,1],grid[1,0],grid[1,1]]\\n combined_order=[\\\'ROC\\\',\\\'Balanced_accuracy\\\',\\\'Success_rate\\\',\\\'PR_AUC\\\']\\n for kind,(fig,ax) in figures.items():\\n  title,xlab,ylab=configs[kind];ax.set(title=title,xlabel=xlab,ylabel=ylab)\\n  ax.set_xlim(0,100 if kind==\\\'Success_rate\\\' else 1)\\n  ax.set_ylim(0,102 if kind==\\\'Success_rate\\\' else 1.025)\\n  ax.xaxis.set_major_locator(MultipleLocator(20 if kind==\\\'Success_rate\\\' else .2))\\n  ax.yaxis.set_major_locator(MultipleLocator(20 if kind==\\\'Success_rate\\\' else .2))\\n  if kind==\\\'ROC\\\':ax.plot([0,1],[0,1],color=\\\'#87929C\\\',lw=1,ls=\\\':\\\',zorder=0)\\n  if kind==\\\'Success_rate\\\':ax.plot([0,100],[0,100],color=\\\'#87929C\\\',lw=1,ls=\\\':\\\',zorder=0)\\n  if kind==\\\'Balanced_accuracy\\\':\\n   ax.axvline(.5,color=\\\'#87929C\\\',lw=1,ls=\\\':\\\',zorder=0)\\n   ax.axhline(.5,color=\\\'#B7BEC5\\\',lw=.8,ls=\\\':\\\',zorder=0)\\n  ax.legend(loc=\\\'upper center\\\',bbox_to_anchor=(.5,-.19),ncol=2,frameon=False,handlelength=3,columnspacing=1.5,labelspacing=.8)\\n  for ext in [\\\'jpg\\\',\\\'svg\\\']:fig.savefig(output/(kind+\\\'.\\\'+ext),dpi=350)\\n  # Copy plotted data and styles, never rasterize the single-panel images.\\n  position=combined_order.index(kind)\\n  target=combined.add_subplot(slots[position])\\n  for line in ax.lines:\\n   target.plot(line.get_xdata(),line.get_ydata(),color=line.get_color(),\\n               linestyle=line.get_linestyle(),linewidth=1.9 if line.get_linewidth()>1 else .85,\\n               alpha=line.get_alpha(),label=line.get_label())\\n  target.set(xscale=ax.get_xscale(),yscale=ax.get_yscale(),xlim=ax.get_xlim(),ylim=ax.get_ylim(),\\n             xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel())\\n  target.set_title(title,fontsize=15,pad=13)\\n  target.xaxis.label.set_size(11);target.yaxis.label.set_size(11)\\n  target.tick_params(labelsize=10)\\n  style_axes(target)\\n  target.grid(color=\\\'#E8EBEF\\\',linewidth=.6);target.set_axisbelow(True)\\n  target.text(-.105,1.035,f\\\'({chr(97+position)})\\\',transform=target.transAxes,\\n              fontsize=15,fontweight=\\\'bold\\\',va=\\\'bottom\\\')\\n  if kind in [\\\'ROC\\\',\\\'PR_AUC\\\',\\\'Balanced_accuracy\\\']:\\n   field={\\\'ROC\\\':\\\'ROC_AUC\\\',\\\'PR_AUC\\\':\\\'PR_AUC_trapezoidal\\\',\\\'Balanced_accuracy\\\':\\\'balanced_accuracy_at_0_5\\\'}[kind]\\n   heading={\\\'ROC\\\':\\\'AUC\\\',\\\'PR_AUC\\\':\\\'PR-AUC\\\',\\\'Balanced_accuracy\\\':\\\'BA at 0.5\\\'}[kind]\\n   annotations=\\\'\\\\n\\\'.join(f\\\'{label}: {row[field]:.4f}\\\' for label,row in zip(LABELS,rows))\\n   target.text(.97,.06 if kind!=\\\'PR_AUC\\\' else .32,heading+\\\'\\\\n\\\'+annotations,\\n               transform=target.transAxes,ha=\\\'right\\\',va=\\\'bottom\\\',fontsize=9,\\n               linespacing=1.4,bbox=dict(boxstyle=\\\'round,pad=0.6,rounding_size=0.15\\\',facecolor=\\\'white\\\',alpha=.96,edgecolor=\\\'#D2D8DF\\\',linewidth=.8))\\n  plt.close(fig)\\n handles=[Line2D([0],[0],color=color,lw=2,ls=\\\'--\\\' if i%2 else \\\'-\\\',label=label)\\n          for i,(label,color) in enumerate(zip(LABELS,COLORS))]\\n combined.legend(handles=handles,loc=\\\'lower center\\\',bbox_to_anchor=(.5,.025),\\n                 ncol=3,frameon=False,fontsize=12,handlelength=3,columnspacing=2.5,labelspacing=.9)\\n for ext in [\\\'jpg\\\',\\\'svg\\\']:combined.savefig(output/(\\\'All_four_metrics.\\\'+ext),dpi=300)\\n plt.close(combined)\\n pd.DataFrame(rows).to_csv(output/\\\'metrics.csv\\\',index=False)\\n (output/\\\'README.txt\\\').write_text(\\\'Four panels: ROC, balanced accuracy, success rate, precision-recall. Lift omitted because it is derived from success rate and extreme small-sample values reduce readability. ROC-AUC independently verified on current-model evaluation samples. Original/PCA share family colours with solid/dashed lines. PR-AUC is trapezoidal; AP separately reported. BA threshold=0.5. Success uses ROC evaluation units: labelled pixels for RF/1DCNN, 9x9 patches for 2DCNN. X is sample fraction, not geographic area. Different populations and prevalences limit comparison, especially PR-AUC. Legacy descriptive evaluation, not independent validation. No smoothing or changes to scores.\\\\n\\\',encoding=\\\'utf8\\\')\\n print(pd.DataFrame(rows).to_string(index=False))\\n\\nif __name__==\\\'__main__\\\':\\n p=argparse.ArgumentParser();p.add_argument(\\\'--input\\\',type=Path,default=BASE/\\\'PAC_manuscript_results\\\');p.add_argument(\\\'--output\\\',type=Path,default=BASE/\\\'PAC_four_figures\\\');a=p.parse_args();main(a.input,a.output)\\n\'\n\nfrom pathlib import Path\nimport argparse,json,hashlib,shutil,gc\nimport numpy as np,pandas as pd\nnp.trapezoid=getattr(np,\'trapezoid\',np.trapz)\n\ndef extract_features(model,kind,data,cols,folder):\n import tensorflow as tf\n stats=[]\n for label in [0,1]:\n  d=data.loc[data.Label==label,cols].describe().T.reset_index(names=\'feature\');d.insert(0,\'Label\',label);stats.append(d)\n pd.concat(stats).to_csv(folder/\'input_feature_statistics_by_label.csv\',index=False)\n corr=data[cols].corr();corr.to_csv(folder/\'input_feature_correlations.csv\');fig,ax=e.plt.subplots(figsize=(11,9));im=ax.imshow(corr,vmin=-1,vmax=1,cmap=\'coolwarm\');ax.set_xticks(range(23),cols,rotation=90,fontsize=7);ax.set_yticks(range(23),cols,fontsize=7);fig.colorbar(im,ax=ax,label=\'Pearson correlation (training inputs)\');fig.tight_layout();e.save_pair(fig,folder,\'input_feature_correlations\')\n data.to_csv(folder/\'training_input_features_and_labels.csv\',index=False)\n if kind==\'RF\':return\n X,y=e.legacy_tensors(data,cols,kind);T=X.reshape(-1,23,1) if kind==\'1DCNN\' else X;encoder=tf.keras.Model(inputs=model.inputs,outputs=model.layers[-2].output);latent=encoder.predict(T,batch_size=8192 if kind==\'1DCNN\' else 256,verbose=0);assert len(latent)==len(y);np.savez_compressed(folder/\'CNN_training_latent_features.npz\',features=latent,Label=y,layer_name=model.layers[-2].name,shape=latent.shape)\n if kind==\'1DCNN\':index=data[[\'x\',\'y\',\'Label\']].copy();index.insert(0,\'latent_row\',range(len(index)))\n else:\n  members=[];index=[];tid=0\n  for label in [1,0]:\n   ids=np.flatnonzero(data.Label.to_numpy()==label);n=len(ids)//81;ids=ids[:n*81];d=data.iloc[ids][[\'x\',\'y\',\'Label\']].copy();d.insert(0,\'training_row\',ids);d.insert(0,\'slot\',np.tile(np.arange(81),n));d.insert(0,\'latent_row\',np.repeat(np.arange(tid,tid+n),81));members.append(d);index.extend(dict(latent_row=j,Label=label,source_rows=81,geographic_neighborhood=False) for j in range(tid,tid+n));tid+=n\n  pd.concat(members).to_csv(folder/\'CNN_2D_tensor_source_rows.csv\',index=False);index=pd.DataFrame(index)\n index.to_csv(folder/\'CNN_latent_row_index.csv\',index=False);e.write_json(folder/\'CNN_latent_features_method.json\',dict(layer=model.layers[-2].name,shape=list(latent.shape),population=\'actual training inputs in original order/legacy tensor packing\',interpretation=\'learned pre-output representation, not PCA eigenvalues or causal feature effects\'))\n\ndef feature_panels(co):\n fig,axes=e.plt.subplots(3,2,figsize=(20,16));fig.subplots_adjust(hspace=.65,wspace=.25,bottom=.12)\n for (b,k),ax in zip(e.ORDER,axes.ravel()):\n  d=pd.read_csv(co/(b+\'_\'+k)/\'feature_contributions.csv\');ax.bar(range(23),d.importance,color=\'#2166ac\' if b==\'Raw\' else \'#b35806\');ax.set_xticks(range(23),d.feature,rotation=70,ha=\'right\',fontsize=7);ax.set(title=b+\' \'+k,ylabel=\'Gini importance\' if k==\'RF\' else \'Mean |probability change|\')\n fig.suptitle(\'Feature contributions: RF Gini / CNN zero-channel ablation (different scales)\',fontsize=15);e.save_pair(fig,co,\'All_six_feature_contributions\')\n\ndef pca_eigenvalue_audit(root,out):\n folder=out/\'PCA_eigenvalues\';folder.mkdir(exist_ok=True);cols=[f\'PCA{i}\' for i in range(1,24)];n=0;mean=np.zeros(23);m2=np.zeros(23)\n for frame in pd.read_csv(root/\'data/data/PCA_with_labels_Test.csv\',usecols=cols,chunksize=100000):\n  z=frame[cols].to_numpy();cn=len(z);cm=z.mean(axis=0);cs=((z-cm)**2).sum(axis=0);delta=cm-mean;total=n+cn;m2+=cs+delta**2*n*cn/total;mean+=delta*cn/total;n=total\n eigen=m2/(n-1);arch=root/\'data/PCA_test/PCA_test/PCA_variance_contributions.xlsx\';old=pd.read_excel(arch);old.to_csv(folder/\'archived_original_unsorted_rows.csv\',index=False);shutil.copy2(arch,folder/\'archived_original_variance_table.xlsx\');sorted_old=np.sort(old.Eigenvalues.to_numpy())[::-1];difference=float(np.max(abs(eigen-sorted_old)));assert difference<1e-8,\'Archived spectrum does not match current PC data; do not relabel it silently\'\n ratio=eigen/eigen.sum();table=pd.DataFrame(dict(feature=cols,eigenvalue_current_PC_variance=eigen,explained_variance_ratio=ratio,cumulative_variance_percent=np.cumsum(ratio)*100,archived_eigenvalue_sorted=sorted_old,absolute_difference=abs(eigen-sorted_old)));table.to_csv(folder/\'PCA_eigenvalues_verified_current_order.csv\',index=False)\n fig,ax=e.plt.subplots(figsize=(12,5));right=ax.twinx();ax.bar(range(1,24),eigen,color=\'steelblue\',label=\'Eigenvalue\');right.plot(range(1,24),ratio.cumsum()*100,\'o-\',color=\'firebrick\',label=\'Cumulative variance\');ax.set_xticks(range(1,24),cols,rotation=65,ha=\'right\');ax.set(ylabel=\'Eigenvalue / PC score variance\',xlabel=\'Actual frozen PC column order\',title=\'Verified PCA spectrum — no PCA refitting\');right.set(ylabel=\'Cumulative explained variance (%)\',ylim=(0,105));fig.tight_layout();e.save_pair(fig,folder,\'PCA_scree_verified\')\n e.write_json(folder/\'audit.json\',dict(status=\'verified\',score_rows=n,original_workbook=str(arch),source_sha256=e.sha(arch),original_rows_in_descending_order=bool(np.all(np.diff(old.Eigenvalues)<=0)),current_PC_order_matches_sorted_archived_spectrum=True,max_absolute_difference=difference,PCA_refitted=False,model_inputs_changed=False,explanation=\'Archived np.linalg.eig output was not sorted before labels PCA1..23 were assigned; current sklearn PC columns follow descending variance. Corrected table uses current frozen scores and checks sorted archive eigenvalues; original table preserved. No PCA loading matrix inferred.\'))\n\ndef main():\n import tensorflow as tf\n p=argparse.ArgumentParser(description=__doc__);p.add_argument(\'--root\',type=Path,default=e.ROOT_DEFAULT);p.add_argument(\'--base\',type=Path,default=e.ROOT_DEFAULT/\'fig/New_negative_sample\');p.add_argument(\'--output\',type=Path);p.add_argument(\'--resume\',action=\'store_true\');a=p.parse_args();out=(a.output or a.base/\'Post_analysis_complete\').resolve();out.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,out/\'reproduction_code_snapshot.py\');e.write_json(out/\'COMPLETE.json\',dict(status=\'running\',models=42));pa=e.helper(PA_SOURCE,\'current_user_PA\');pub=e.helper(PUBLICATION_SOURCE,\'current_user_four\');points=pd.read_csv(a.base/\'Final_AB/deposit_records.csv\');points.to_csv(out/\'deposit_records.csv\',index=False)\n provenance=out/\'reference_code\';provenance.mkdir(exist_ok=True);(provenance/\'PA_saved_six_models.embedded.py\').write_text(PA_SOURCE,encoding=\'utf8\');(provenance/\'PAC_publication_four.adapted.py\').write_text(PUBLICATION_SOURCE,encoding=\'utf8\')\n cases=[(\'Final_A\',a.base/\'Final_AB/A\'),(\'Final_B\',a.base/\'Final_AB/B\')]+[(f\'Repeat_{i}\',a.base/f\'A_five_random_repeats/Run_{i}\') for i in range(1,6)];allmetrics=[];allpa=[];allfeatures=[];auditrows=[];completed=0\n for case,source in cases:\n  co=out/case;co.mkdir(exist_ok=True);audit=co/\'ROC_audit\';audit.mkdir(exist_ok=True);metrics=[];pars=[]\n  for branch in [\'Raw\',\'PCA\']:\n   cols=e.cols_for(branch);data=pd.read_csv(source/(branch+\'_training_input.csv\'));full=pd.read_csv(a.root/\'data\'/(\'original_analyses/grayscale_data_combined_test.csv\' if branch==\'Raw\' else \'data/PCA_with_labels_Test.csv\')).sort_values([\'y\',\'x\']).reset_index(drop=True);X=full[cols].to_numpy(np.float32);patches=X.reshape(564,1800,23)[:558].reshape(62,9,200,9,23).transpose(0,2,1,3,4).reshape(-1,9,9,23);labels=np.full(len(full),np.nan);labels[data.y.to_numpy(int)*1800+data.x.to_numpy(int)]=data.Label\n   for kind in [\'RF\',\'1DCNN\',\'2DCNN\']:\n    name=branch+\'_\'+kind;src=source/name;folder=co/name;folder.mkdir(exist_ok=True);weight=src/(\'model.joblib\' if kind==\'RF\' else \'model.keras\');weightsha=e.sha(weight);predsha=e.sha(src/\'full_map_predictions.npz\');done=folder/\'ANALYSIS_COMPLETE.json\';print(\'ANALYSE\',case,name,flush=True)\n    skip=False\n    if a.resume and done.exists():\n     old=json.loads(done.read_text(encoding=\'utf8\'));skip=old[\'weights_sha256\']==weightsha and old[\'source_prediction_sha256\']==predsha\n    if not skip:\n     model=e.joblib.load(weight) if kind==\'RF\' else tf.keras.models.load_model(weight,compile=False);score,ps=e.infer_map(model,kind,X,patches);ref=np.load(src/\'full_map_predictions.npz\')[\'score\'];difference=float(np.nanmax(abs(score-ref)));assert difference<=1e-5,(case,name,difference)\n     row=e.evaluate_map(labels,score,ps,kind,e.CANON[branch,kind],folder,audit,full);row.update(case=case,branch=branch);e.write_json(folder/\'descriptive_metrics.json\',row)\n     frame=full[[\'x\',\'y\']].copy();frame[\'Predicted_Probability\']=score;pdout=folder/\'PA_by_deposit_records\';pdout.mkdir(exist_ok=True);pr=pa.analyse(frame,points,pdout,name);pr.update(case=case,branch=branch);e.write_json(folder/\'PA_metrics.json\',pr)\n     # Reuse complete, matching frozen-weight ablation if available; new repeats have none.\n     cached=(src/\'feature_contributions.csv\').exists() and (src/\'feature_method.json\').exists()\n     cfg=json.loads((src/\'parameters_and_provenance.json\').read_text(encoding=\'utf8\'));assert cfg[\'weights_sha256\']==weightsha\n     if cached:\n      for n in [\'feature_contributions.csv\',\'feature_method.json\',\'feature_contributions.jpg\',\'feature_contributions.svg\']:shutil.copy2(src/n,folder/n)\n     else:e.feature_analysis(model,kind,X,patches,score if kind!=\'2DCNN\' else ps,folder,cols,False)\n     for n in [\'training_history.csv\',\'training_fit_samples.csv\',\'parameters_and_provenance.json\']:\n      if (src/n).exists():shutil.copy2(src/n,folder/n)\n     e.history_figures(folder);extract_features(model,kind,data,cols,folder)\n     fit=pd.read_csv(src/\'training_fit_samples.csv\');fitmetrics=e.metric_row(fit.Label.to_numpy(int),fit.score.to_numpy());e.write_json(folder/\'training_fit_metrics.json\',fitmetrics)\n     np.savez_compressed(folder/\'verified_current_predictions.npz\',score=score.astype(np.float32),shape=[564,1800]);e.write_json(done,dict(status=\'complete\',weights_sha256=weightsha,source_prediction_sha256=predsha,max_prediction_replay_difference=difference,weights_unchanged=e.sha(weight)==weightsha,feature_analysis=\'verified existing same-weight full-map analysis\' if cached else \'new full-map feature analysis\',no_training=True));del model,score,ps,ref,frame;tf.keras.backend.clear_session();gc.collect()\n    row=json.loads((folder/\'descriptive_metrics.json\').read_text(encoding=\'utf8\'));pr=json.loads((folder/\'PA_metrics.json\').read_text(encoding=\'utf8\'));metrics.append(row);pars.append(pr);v=pd.read_csv(folder/\'feature_contributions.csv\');v[\'case\']=case;v[\'branch\']=branch;v[\'model\']=kind;v[\'method\']=\'Gini\' if kind==\'RF\' else \'zero-channel full-map mean absolute change\';allfeatures.append(v);auditrows.append(dict(case=case,model=name,**json.loads(done.read_text(encoding=\'utf8\'))));completed+=1;e.write_json(out/\'progress.json\',dict(case=case,model=name,completed=completed,total=42))\n   del X,patches,full,data;gc.collect()\n  pd.DataFrame(metrics).to_csv(audit/\'metrics_and_ROC_audit.csv\',index=False);pd.DataFrame(pars).to_csv(co/\'PA_deposit_summary.csv\',index=False);pub.main(audit,co/\'four_metrics\');feature_panels(co);allmetrics.extend(metrics);allpa.extend(pars)\n pca_eigenvalue_audit(a.root,out)\n M=pd.DataFrame(allmetrics);P=pd.DataFrame(allpa);F=pd.concat(allfeatures,ignore_index=True);M.to_csv(out/\'all_42_model_metrics.csv\',index=False);P.to_csv(out/\'all_42_PA_metrics.csv\',index=False);F.to_csv(out/\'all_feature_contributions.csv\',index=False);pd.DataFrame(auditrows).to_csv(out/\'weights_predictions_audit.csv\',index=False)\n R=M[M.case.str.startswith(\'Repeat_\')];keys=[\'ROC_AUC\',\'PR_AUC\',\'AP\',\'accuracy\',\'balanced_accuracy\',\'precision\',\'recall\',\'F1\',\'Brier\'];R.groupby([\'branch\',\'model\'])[keys].agg([\'mean\',\'std\']).to_csv(out/\'five_repeats_descriptive_metrics_mean_std.csv\');RF=F[F.case.str.startswith(\'Repeat_\')];RF.groupby([\'branch\',\'model\',\'feature\',\'method\']).importance.agg([\'mean\',\'std\',\'min\',\'max\']).to_csv(out/\'five_repeats_feature_contributions_mean_std.csv\')\n for b,k in e.ORDER:\n  d=RF[(RF.branch==b)&(RF.model==k)].groupby(\'feature\').importance.agg([\'mean\',\'std\']).reindex(e.cols_for(b));fig,ax=e.plt.subplots(figsize=(12,5));ax.bar(np.arange(23),d[\'mean\'],yerr=d[\'std\'],capsize=2,color=\'skyblue\',edgecolor=\'black\');ax.set_xticks(range(23),d.index,rotation=65,ha=\'right\');ax.set(title=b+\' \'+k+\' — five negative-sampling repeats: mean ± sample SD\',ylabel=\'Gini importance\' if k==\'RF\' else \'Mean absolute probability change\');fig.tight_layout();e.save_pair(fig,out,b+\'_\'+k+\'_feature_mean_std\')\n (out/\'README.md\').write_text("""# 42个已训练模型的完整后续分析\n\n范围：Final_AB的A/B各六模型，加A五次随机重复各六模型，共42个。没有训练、调参、重抽负样本或修改原目录/模型。先用权重重新推理，核对已有全区概率，再计算当前评价样本上的指标。\n\n每个case：four_metrics/All_four_metrics.jpg/svg及ROC、BA、成功率、PR单图；ROC_audit导出当前标签/分数与完整曲线；PA_deposit_summary及六模型特征贡献对照图。\n每模型：ROC/AUC、AP与梯形PR-AUC、准确率/BA/精确率/召回率/F1/Brier、混淆矩阵、两种P–A（矿点记录与正评价单位）、训练曲线/训练拟合指标、RF Gini或CNN逐特征置零敏感性、23输入特征的分标签统计/相关矩阵及原训练特征表。\nCNN_training_latent_features.npz保存1D Flatten或2D GAP在实际训练输入上的隐层向量，CNN_latent_row_index.csv对应行索引。2D另导出每张量81行来源，保留原非空间拼接；末72正像元按原代码丢弃。这是隐层表示，不能称为PCA特征值/原始元素贡献。\n\n特征解释：RF Gini和CNN置零引起的概率平均绝对变化尺度不同，不能直接比较大小；置零可能离开训练分布，不是SHAP或因果效应。PCA特征为原始冻结23个PC通道，不重新拟合PCA，也未在无载荷矩阵情况下将PC贡献转换为元素贡献。PCA_eigenvalues目录另保存与当前PC顺序核对的特征值/方差贡献率/碎石图；原存档未排序的特征值表也完整保留，详见audit.json。\n\nPA_saved_six_models.py的analyse原样嵌入，用当前预测代替它原main里的旧RAR预测。PAC_publication_four.py保持计算/布局，将旧论文AUC核对改为当前样本独立重算AUC核对，标题注明描述性，只输出JPG/SVG；嵌入源码见reference_code。\n\nRF/1D ROC使用训练标记像元，2D使用12400个真实制图窗口（任一正像元为正，其余包括未标记窗口当背景）。它们是兼容口径描述性评价，不是独立测试或空间CV。两种评价单位/患病率不同，PR-AUC不可简单横比。P–A面积为等像元比例，不是纬度面积加权。2D北侧6行缺预测，不插补伪造。\n\n五次mean/std文件是负抽样重复间统计，不是CV折间泛化误差。Final_A/B不混入五次统计。均值预测图不是单个训练模型，本次42模型分析不将均值图充当第43个模型。\n\n运行本程序需原project/data全区特征及New_negative_sample下Final_AB和A_five_random_repeats；--resume跳过本轮已完成且权重/预测SHA匹配的模型。原模型/输入/预测保持不变。\n""",encoding=\'utf8\')\n e.write_json(out/\'COMPLETE.json\',dict(status=\'complete\',models=42,new_training=False,all_weights_and_predictions_verified=True,CNN_latent_models=28,full_feature_analysis=True,code_sha256=e.sha(Path(__file__))));print(\'COMPLETE\',out,flush=True)\nif __name__==\'__main__\':main()\n', 'reviewer4_feature_importance': '"""Reviewer 4: archived full-map sensitivities and held-out spatial importance."""\nimport argparse\nfrom importlib.metadata import version\nfrom itertools import combinations\nimport json\nfrom pathlib import Path\n\nfrom spatial_two_schemes import ROOT, RAW, patches\nimport joblib\nimport numpy as np\nimport pandas as pd\nfrom scipy.stats import spearmanr\nfrom sklearn.metrics import roc_auc_score\nimport tensorflow as tf\n\nBASE = Path(__file__).resolve().parent\nMODELS = {\n    \'Raw\': (\'Revised/fig/original-RF_trained_model.joblib\',\n            \'original_analyses/1DCNN/trained_model_origin_1DCNN.h5\',\n            \'original_analyses/2DCNN_model_original_analyse.h5\'),\n    \'PCA\': (\'Revised/fig/PCA-RF_trained_model.joblib\',\n            \'1DCNN_test/ensure/trained_model_1DCNN_test.h5\',\n            \'2DCNN_test/ensure/2DCNN_model.h5\'),\n}\nKINDS = (\'RF\', \'1DCNN\', \'2DCNN\')\n\n\ndef scores(model, kind, data):\n    if kind == \'RF\':\n        return model.predict_proba(data.reshape(-1, 23))[:, 1]\n    shape = (-1, 23, 1) if kind == \'1DCNN\' else (-1, 9, 9, 23)\n    inputs = data.reshape(shape)\n    return np.concatenate([\n        model(batch, training=False).numpy().ravel()\n        for batch in np.array_split(inputs, max(1, (len(inputs) + 4095) // 4096))\n    ])\n\n\ndef site_scores(model, kind, data):\n    return scores(model, kind, data).reshape(len(data), -1).mean(axis=1)\n\n\ndef load(path, kind):\n    if kind == \'RF\':\n        model = joblib.load(path)\n        model.n_jobs = 4\n        return model\n    return tf.keras.models.load_model(path, compile=False)\n\n\ndef spatial(args):\n    sites = pd.read_csv(args.cv / \'fixed_sites.csv\')\n    records = []\n    for branch in MODELS:\n        cols, grid = read_grid(args, branch)\n        for kind in KINDS:\n            for k in range(5):\n                folder = args.cv / f\'{branch}_{kind}\' / f\'fold_{k+1}\'\n                idx = np.flatnonzero(sites.fold == k)\n                test = sites.iloc[idx]\n                data = patches(grid, test[[\'x\', \'y\']].to_numpy())\n                y = test.label.to_numpy()\n                model = load(folder / (\'model.joblib\' if kind == \'RF\' else \'model.keras\'), kind)\n                base = site_scores(model, kind, data)\n                cached = pd.read_csv(folder / \'heldout.csv\')\n                np.testing.assert_array_equal(cached.site, idx)\n                np.testing.assert_array_equal(cached.label, y)\n                np.testing.assert_allclose(cached.score, base, atol=1e-5, rtol=0)\n                auc = roc_auc_score(y, base)\n                rng = np.random.default_rng(args.seed + k)\n                orders = np.stack([rng.permutation(len(y)) for _ in range(args.repeats)])\n                for j, feat in enumerate(cols):\n                    for start in range(0, args.repeats, 5):\n                        order = orders[start:start+5]\n                        perm = np.repeat(data[None], len(order), axis=0)\n                        perm[..., j] = data[order, :, :, j]\n                        changed = site_scores(model, kind, perm.reshape(-1, 9, 9, 23))\n                        for r, values in enumerate(changed.reshape(len(order), -1), start+1):\n                            records.append((branch, kind, k+1, feat, r, auc,\n                                            auc - roc_auc_score(y, values)))\n                print(branch, kind, k+1, flush=True)\n                del model\n                tf.keras.backend.clear_session()\n                pd.DataFrame(records, columns=[\'branch\',\'model\',\'fold\',\'feature\',\'repeat\',\n                                               \'baseline_auc\',\'delta_auc\']).to_csv(\n                                                   args.out / \'permutation_repeats.csv\', index=False)\n    summarize(args, pd.read_csv(args.out / \'permutation_repeats.csv\'))\n\n\ndef summarize(args, frame):\n    expected = len(MODELS) * len(KINDS) * 5 * 23 * args.repeats\n    identifiers = [\'branch\', \'model\', \'fold\', \'feature\', \'repeat\']\n    if len(frame) != expected or frame.duplicated(identifiers).any():\n        raise ValueError(\'Incomplete or duplicate spatial permutation results\')\n    keys = [\'branch\', \'model\', \'feature\']\n    folds = frame.groupby(keys + [\'fold\']).agg(\n        delta_mean=(\'delta_auc\',\'mean\'), repeat_sd=(\'delta_auc\',\'std\'),\n        baseline_auc=(\'baseline_auc\',\'first\')).reset_index()\n    folds[\'rank\'] = folds.groupby([\'branch\',\'model\',\'fold\']).delta_mean.rank(\n        ascending=False, method=\'average\')\n    folds[\'top5\'] = folds.delta_mean >= folds.groupby(\n        [\'branch\',\'model\',\'fold\']).delta_mean.transform(lambda x: x.nlargest(5).iloc[-1])\n    folds.to_csv(args.out / \'fold_importance.csv\', index=False)\n    summary = folds.groupby(keys).agg(\n        mean_delta_auc=(\'delta_mean\',\'mean\'), fold_sd=(\'delta_mean\',\'std\'),\n        mean_repeat_sd=(\'repeat_sd\',\'mean\'), mean_rank=(\'rank\',\'mean\'),\n        min_rank=(\'rank\',\'min\'), max_rank=(\'rank\',\'max\'),\n        top5_folds=(\'top5\',\'sum\')).reset_index()\n    summary.sort_values([\'branch\',\'model\',\'mean_delta_auc\'], ascending=[True,True,False]).to_csv(\n        args.out / \'importance_summary.csv\', index=False)\n    pairs = []\n    for (branch, kind), group in folds.groupby([\'branch\',\'model\']):\n        ranks = group.pivot(index=\'feature\', columns=\'fold\', values=\'rank\')\n        top = group.pivot(index=\'feature\', columns=\'fold\', values=\'top5\')\n        for a,b in combinations(ranks.columns, 2):\n            rho = spearmanr(ranks[a], ranks[b]).statistic\n            jaccard = (top[a] & top[b]).sum() / (top[a] | top[b]).sum()\n            pairs.append((branch,kind,a,b,rho,jaccard))\n    pd.DataFrame(pairs, columns=[\'branch\',\'model\',\'fold_a\',\'fold_b\',\'spearman\',\'top5_jaccard\']).to_csv(\n        args.out / \'rank_agreement.csv\', index=False)\n    side_by_side(summary, \'mean_delta_auc\', args.out / \'spatial_rankings_side_by_side.csv\')\n\n\ndef side_by_side(frame, score, destination):\n    ranked = frame.sort_values([\'model\', \'branch\', score, \'feature\'],\n                               ascending=[True, True, False, True]).copy()\n    ranked[\'position\'] = ranked.groupby([\'model\', \'branch\']).cumcount() + 1\n    left, right = (ranked[ranked.branch == branch].drop(columns=\'branch\') for branch in MODELS)\n    comparison = left.merge(right, on=[\'model\', \'position\'], suffixes=(\'_Original\', \'_PCA\'),\n                            validate=\'one_to_one\')\n    comparison.to_csv(destination, index=False)\n\n\ndef read_grid(args, branch):\n    rel = (\'original_analyses/grayscale_data_combined_test.csv\' if branch == \'Raw\'\n           else \'data/PCA_with_labels_Test.csv\')\n    cols = RAW if branch == \'Raw\' else [f\'PCA{i}\' for i in range(1,24)]\n    frame = pd.read_csv(args.root / \'data\' / rel, usecols=[\'x\',\'y\',*cols]).sort_values([\'y\',\'x\'])\n    np.testing.assert_array_equal(frame.x, np.tile(np.arange(1800),564))\n    np.testing.assert_array_equal(frame.y, np.repeat(np.arange(564),1800))\n    grid = frame[cols].to_numpy(np.float32).reshape(564,1800,23)\n    if not np.isfinite(grid).all():\n        raise ValueError(f\'{branch}: nonfinite predictors\')\n    return cols, grid\n\n\ndef archived(args):\n    rows = []\n    for branch, members in MODELS.items():\n        cols, grid = read_grid(args, branch)\n        for kind, member in zip(KINDS, members):\n            model = load(args.models / member, kind)\n            if kind == \'RF\':\n                means = model.feature_importances_\n                sd = np.std([t.feature_importances_ for t in model.estimators_], axis=0, ddof=1)\n                n = len(model.estimators_)\n                unit = \'tree\'\n                metric = \'normalized MDI\'\n            else:\n                if kind == \'1DCNN\':\n                    data = grid.reshape(-1,23)\n                    axis = \'pixel\'\n                else:\n                    xx, yy = np.meshgrid(np.arange(4,1796,9), np.arange(4,560,9))\n                    data = patches(grid, np.column_stack([xx.ravel(), yy.ravel()]))\n                    axis = \'9x9 window\'\n                baseline = scores(model, kind, data)\n                means, sd = [], []\n                for j in range(23):\n                    changed = data.copy()\n                    changed[...,j] = 0\n                    diff = np.abs(baseline - scores(model, kind, changed))\n                    means.append(diff.mean())\n                    sd.append(diff.std(ddof=1))\n                n = len(baseline)\n                unit = axis\n                metric = \'mean absolute probability change\'\n            for feat, mean, deviation, rank in zip(cols, means, sd, pd.Series(means).rank(ascending=False)):\n                rows.append((branch,kind,feat,metric,mean,deviation,n,unit,rank))\n            pd.DataFrame(rows, columns=[\'branch\',\'model\',\'feature\',\'metric\',\'score\',\'sd\',\'n\',\n                                        \'sd_unit\',\'rank\']).sort_values([\'model\',\'branch\',\'rank\']).to_csv(\n                                            args.out / \'archived_importance.csv\', index=False)\n            print(\'Archived\',branch,kind,flush=True)\n            del model\n            tf.keras.backend.clear_session()\n    side_by_side(pd.read_csv(args.out / \'archived_importance.csv\'), \'score\',\n                 args.out / \'archived_rankings_side_by_side.csv\')\n\n\ndef main():\n    p = argparse.ArgumentParser(description=__doc__)\n    p.add_argument(\'--root\', type=Path, default=ROOT)\n    p.add_argument(\'--cv\', type=Path, default=BASE/\'fixed_labels_spatial_cv\')\n    p.add_argument(\'--models\', type=Path, default=BASE/\'parameter_audit/archive/central asian mapping\')\n    p.add_argument(\'--out\', type=Path, default=BASE/\'reviewer4_importance\')\n    p.add_argument(\'--repeats\', type=int, default=30)\n    p.add_argument(\'--seed\', type=int, default=4226)\n    p.add_argument(\'--stage\', choices=[\'all\',\'archived\',\'spatial\'], default=\'all\')\n    args = p.parse_args()\n    if args.repeats < 2:\n        p.error(\'--repeats must be at least 2\')\n    args.out.mkdir(parents=True, exist_ok=True)\n    (args.out/\'environment.json\').write_text(json.dumps({\n        name: version(name) for name in (\'numpy\',\'pandas\',\'scipy\',\'scikit-learn\',\n                                         \'joblib\',\'tensorflow\',\'keras\')}, indent=2))\n    if args.stage in (\'all\',\'spatial\'):\n        (args.out/\'permutation_settings.json\').write_text(json.dumps(\n            {\'repeats\': args.repeats, \'seed\': args.seed, \'unit\':\'whole 9x9 channel between held-out sites\',\n             \'score\':\'baseline ROC-AUC minus shuffled ROC-AUC\'}, indent=2))\n        spatial(args)\n    if args.stage in (\'all\',\'archived\'):\n        archived(args)\n\n\nif __name__ == \'__main__\':\n    main()\n', 'plot_reviewer4_figure9': 'from pathlib import Path\nimport os\nimport sys\n\nimport matplotlib\nmatplotlib.use(\'Agg\')\nimport matplotlib.pyplot as plt\nfrom matplotlib.ticker import MaxNLocator\nimport numpy as np\nimport pandas as pd\n\nout=Path(os.environ[\'CAOB_OUTPUT\'])/\'Feature_Importance\'\narch=pd.read_csv(out/\'archived_importance.csv\')\noriginal=\'--original-method\' in sys.argv\nspatial_out=out/\'original_method_spatial\' if original else out\nspatial=pd.read_csv(spatial_out/\'fold_importance.csv\')\nfield=\'score\' if original else \'delta_mean\'\nassert len(arch)==138 and len(spatial)==690\nplt.rcParams.update({\'font.size\':11,\'axes.titleweight\':\'bold\',\'svg.fonttype\':\'none\',\'axes.spines.top\':True,\'axes.spines.right\':True})\nraw=arch[(arch.branch==\'Raw\')&(arch.model==\'RF\')].feature.sort_values().tolist()\nnames={\'RF\':\'RF\',\'1DCNN\':\'1-D CNN\',\'2DCNN\':\'2-D CNN\'}\nextent=spatial.groupby([\'branch\',\'model\',\'feature\'])[field].agg([\'mean\',\'std\'])\ncv_low=min(spatial[field].min(),(extent[\'mean\']-extent[\'std\']).min())-.02\ncv_high=max(spatial[field].max(),(extent[\'mean\']+extent[\'std\']).max())+.02\nfor cv in ((True,) if original else (False,True)):\n    fig,axs=plt.subplots(3,2,figsize=(14,10),layout=\'constrained\',sharex=\'col\')\n    for i,(kind,title) in enumerate(names.items()):\n        for j,branch in enumerate((\'Raw\',\'PCA\')):\n            ax=axs[i,j]\n            features=raw if j==0 else [f\'PCA{k}\' for k in range(1,24)]\n            x=np.arange(23)\n            colors=[\'#8bcce2\']*23\n            if j==1:\n                colors[6]=\'#377eaa\'\n            if cv:\n                tab=spatial[(spatial.branch==branch)&(spatial.model==kind)].pivot(index=\'feature\',columns=\'fold\',values=field).loc[features]\n                y=tab.mean(axis=1)\n                err=tab.std(axis=1)\n                ax.axhline(0,color=\'#555555\',lw=.85,zorder=2)\n            else:\n                y=arch[(arch.branch==branch)&(arch.model==kind)].set_index(\'feature\').loc[features,\'score\']\n                err=None\n            ax.bar(x,y,color=colors,edgecolor=\'#202020\',linewidth=.75,yerr=err,capsize=2,error_kw={\'elinewidth\':.8},zorder=3)\n            if cv:\n                for k,offset in enumerate(np.linspace(-.23,.23,5)):\n                    ax.scatter(x+offset,tab.iloc[:,k],s=8,color=\'#242424\',alpha=.66,zorder=4)\n                if original:\n                    low=min(0,(y-err).min())\n                    high=max((y+err).max(),tab.to_numpy().max())\n                    ax.set_ylim(low,high*1.17)\n                else:\n                    ax.set_ylim(cv_low,cv_high)\n            else:\n                ax.set_ylim(0,y.max()*1.17)\n            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))\n            ax.set_title(f\'{"Original" if j==0 else "PCA"} | {title}\',pad=8)\n            ax.set_ylabel(\'ROC-AUC decrease\' if cv and not original else (\'Gini importance\' if i==0 else \'Mean absolute probability change\'))\n            ax.text(.975,.92,chr(97+i*2+j),transform=ax.transAxes,ha=\'center\',va=\'center\',weight=\'bold\',\n                    bbox=dict(boxstyle=\'circle,pad=.29\',fc=\'white\',ec=\'#222222\'),zorder=6)\n            ax.set_xlim(-.7,22.7)\n            ax.set_xticks(x,features if j==0 else [str(k) for k in range(1,24)],rotation=62 if j==0 else 0,\n                          ha=\'right\' if j==0 else \'center\',fontsize=9)\n            ax.grid(axis=\'y\',alpha=.16,zorder=0)\n    axs[2,0].set_xlabel(\'Original predictor\')\n    axs[2,1].set_xlabel(\'PCA index\')\n    name=\'Figure9_spatial_permutation\' if cv else \'Figure9_original_vs_pca\'\n    title=(\'Spatial permutation importance | Bars: five-fold mean ± SD · Dots: individual folds\'\n           if cv else \'Feature contribution in archived models | Independent y-axis scales\')\n    if original:\n        name=\'Figure9_spatial_original_method\'\n        title=\'Spatial folds | Mean ± SD and individual folds · Independent y-axis scales\'\n    fig.suptitle(title,fontsize=15,fontweight=\'bold\')\n    for ext in (\'png\',\'svg\'):\n        fig.savefig(spatial_out/f\'{name}.{ext}\',dpi=220)\n    plt.close(fig)\n\n\n\n', 'reviewer4_original_method_folds': '"""Use the manuscript\'s original feature metrics on each spatial fold."""\nimport os\nfrom pathlib import Path\nfrom types import SimpleNamespace\nfrom itertools import combinations\n\nimport numpy as np\nimport pandas as pd\nfrom scipy.stats import spearmanr\n\nfrom reviewer4_feature_importance import BASE, ROOT, MODELS, KINDS, read_grid, load, patches, scores, tf, side_by_side\n\nCV = Path(os.environ[\'CAOB_SPATIAL\'])\nOUT = Path(os.environ[\'CAOB_OUTPUT\']) / \'Feature_Importance\' / \'original_method_spatial\'\nOUT.mkdir(exist_ok=True)\n\ndef main():\n    sites = pd.read_csv(CV / \'fixed_sites.csv\')\n    rows = []\n    for branch in MODELS:\n        names, grid = read_grid(SimpleNamespace(root=ROOT), branch)\n        for kind in KINDS:\n            for fold in range(1, 6):\n                folder = CV / f\'{branch}_{kind}\' / f\'fold_{fold}\'\n                model = load(folder / (\'model.joblib\' if kind == \'RF\' else \'model.keras\'), kind)\n                if kind == \'RF\':\n                    values = model.feature_importances_\n                    spread = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0, ddof=1)\n                    count = len(model.estimators_)\n                    unit = \'tree\'\n                else:\n                    xy = sites.loc[sites.fold == fold-1, [\'x\',\'y\']].to_numpy()\n                    data = patches(grid, xy)\n                    base = scores(model, kind, data)\n                    cached = pd.read_csv(folder/\'heldout.csv\').score\n                    np.testing.assert_allclose(base.reshape(len(xy), -1).mean(axis=1), cached, atol=1e-5, rtol=0)\n                    diffs = []\n                    for j in range(len(names)):\n                        changed = data.copy()\n                        changed[..., j] = 0\n                        diffs.append(np.abs(base-scores(model, kind, changed)))\n                    values = np.mean(diffs, axis=1)\n                    spread = np.std(diffs, axis=1, ddof=1)\n                    count = len(base)\n                    unit = \'pixel\' if kind == \'1DCNN\' else \'window\'\n                for name, value, sd in zip(names, values, spread):\n                    rows.append((branch,kind,fold,name,value,sd,count,unit))\n                del model\n                tf.keras.backend.clear_session()\n    f = pd.DataFrame(rows, columns=[\'branch\',\'model\',\'fold\',\'feature\',\'score\',\'within_fold_sd\',\'n\',\'sd_unit\'])\n    assert len(f) == 690 and np.isfinite(f[[\'score\',\'within_fold_sd\']]).all().all()\n    groups = f.groupby([\'branch\',\'model\',\'fold\']).score\n    f[\'rank\'] = groups.rank(ascending=False)\n    f[\'top5\'] = f.score >= groups.transform(lambda x: x.nlargest(5).iloc[-1])\n    f.to_csv(OUT/\'fold_importance.csv\', index=False)\n    summary = f.groupby([\'branch\',\'model\',\'feature\']).agg(\n        mean_score=(\'score\',\'mean\'), fold_sd=(\'score\',\'std\'), mean_rank=(\'rank\',\'mean\'),\n        min_rank=(\'rank\',\'min\'), max_rank=(\'rank\',\'max\'), top5_folds=(\'top5\',\'sum\')).reset_index()\n    side_by_side(summary, \'mean_score\', OUT/\'rankings_side_by_side.csv\')\n    pairs = []\n    for (branch,kind), group in f.groupby([\'branch\',\'model\']):\n        ranks = group.pivot(index=\'feature\',columns=\'fold\',values=\'rank\')\n        top = group.pivot(index=\'feature\',columns=\'fold\',values=\'top5\')\n        for a,b in combinations(ranks.columns,2):\n            pairs.append((branch,kind,a,b,spearmanr(ranks[a],ranks[b]).statistic,\n                          (top[a]&top[b]).sum()/(top[a]|top[b]).sum()))\n    pd.DataFrame(pairs,columns=[\'branch\',\'model\',\'fold_a\',\'fold_b\',\'spearman\',\'top5_jaccard\']).to_csv(\n        OUT/\'rank_agreement.csv\',index=False)\n    print(f[(f.branch==\'PCA\')&(f.feature==\'PCA7\')].to_string(index=False))\n    print(pd.DataFrame(pairs,columns=[\'branch\',\'model\',\'a\',\'b\',\'rho\',\'jaccard\']).groupby(\n        [\'branch\',\'model\'])[[\'rho\',\'jaccard\']].median())\n\nif __name__ == \'__main__\':\n    main()\n'}
ARCHIVED_MEMBERS = ['1DCNN_test/ensure/trained_model_1DCNN_test.h5', '2DCNN_test/ensure/2DCNN_model.h5', '2DCNN_test/ensure/PCA_with_full_predictions_Test_2DCNN_test.csv', 'Revised/fig/PCA-1DCNN_ROC_curve_data.csv', 'Revised/fig/PCA-1DCNN_predictions.csv', 'Revised/fig/PCA-RF_ROC_curve_data.csv', 'Revised/fig/PCA-RF_predictions.csv', 'Revised/fig/PCA-RF_trained_model.joblib', 'Revised/fig/PCA-transformed 2DCNN_ROC_curve_data.csv', 'Revised/fig/original-1DCNN_ROC_curve_data.csv', 'Revised/fig/original-1DCNN_predictions.csv', 'Revised/fig/original-2DCNN_ROC_curve_data.csv', 'Revised/fig/original-2DCNN_predictions.csv', 'Revised/fig/original-RF_ROC_curve_data.csv', 'Revised/fig/original-RF_predictions.csv', 'Revised/fig/original-RF_trained_model.joblib', 'original_analyses/1DCNN/PCA_with_predictions_origin_1DCNN.csv', 'original_analyses/1DCNN/trained_model_origin_1DCNN.h5', 'original_analyses/2DCNN_model_original_analyse.h5']

def frozen_inputs(prefix):
    yield prefix / "Final_AB/deposit_records.csv"
    cases = ["Final_AB/A", "Final_AB/B"] + [
        f"A_five_random_repeats/Run_{number}" for number in range(1, 6)
    ]
    for case in cases:
        folder = prefix / case
        for branch in ("Raw", "PCA"):
            yield folder / f"{branch}_training_input.csv"
            for kind in ("RF", "1DCNN", "2DCNN"):
                model = folder / f"{branch}_{kind}"
                yield model / ("model.joblib" if kind == "RF" else "model.keras")
                for name in ("parameters_and_provenance.json", "full_map_predictions.npz",
                             "training_fit_samples.csv"):
                    yield model / name


def repeat_inputs(prefix):
    for number in range(1, 11):
        folder = prefix / f"run_{number:02}"
        yield folder / "ROC__AUC_summary.csv"
        for name in ("original-RF", "PCA-RF", "original-1DCNN",
                     "PCA-1DCNN", "original-2DCNN"):
            yield folder / f"results/Revised/fig/{name}_predictions.csv"
        yield folder / "results/data/PCA_with_full_predictions_Test_2DCNN.csv"
        yield folder / "results/original_analyses/1DCNN/PCA_with_predictions_origin_1DCNN.csv"



def organize_original_outputs(folder):
    """Group the legacy flat exports after all legacy consumers have finished."""
    groups = {
        "ARCHIVED_EPOCH": "Training_Curves",
        "SAVED": "Verified_Inputs",
        "PCA_SAVED": "PCA_Maps",
        "PCA_REFIT": "PCA_Refit",
        "PCA_1DCNN_FROZEN": "PCA_1DCNN",
        "PCA_2DCNN_FROZEN": "PCA_2DCNN",
        "PCA_RF_REFIT": "PCA_RF",
        "RF_REFIT": "Original_RF",
        "Original_RF": "Original_RF",
        "Original_1DCNN": "Original_1DCNN",
        "PA": "PA",
        "ROC": "ROC_AUC",
    }
    models = {
        "original_001": "PCA_1DCNN", "original_003": "PCA_2DCNN",
        "original_056": "Original_2DCNN", "original_072": "Original_1DCNN",
        "original_073": "Original_2DCNN", "original_074": "Original_RF",
        "original_075": "PCA_1DCNN", "original_076": "PCA_2DCNN",
        "original_077": "PCA_RF", "original_079": "PCA_RF",
        "original_080": "Sample_Locations",
    }
    paths = {}
    for file in sorted(folder.iterdir()):
        if not file.is_file():
            continue
        prefix = file.name.split("__", 1)[0]
        category = groups.get(prefix) or models.get(prefix) or "Run_Records"
        target = folder / category / file.name
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        file.rename(target)
        paths[file.name] = target.relative_to(folder).as_posix()
    (folder / "file_index.json").write_text(json.dumps(paths, indent=2), encoding="utf8")


def review_main():
    import argparse
    import importlib.metadata
    import platform
    import re
    import subprocess
    import sys
    import tempfile
    from datetime import datetime

    stages = ("original", "publication", "permutation", "rf_spatial",
              "fixed_spatial", "importance", "repeat10", "frozen42")
    parser = argparse.ArgumentParser(description="CAOB 原论文与审稿补充分析")
    parser.add_argument("--project", type=Path, default=DATA_ROOT.parent)
    parser.add_argument("--only", nargs="+", choices=stages, default=stages)
    parser.add_argument("--check", action="store_true", help="只检查所选阶段输入")
    parser.add_argument("--label", help="在 result 下使用指定的新目录名")
    args = parser.parse_args()
    project = args.project.resolve()
    data = project / "data"
    extra = data / "review_reproduction"
    archive = extra / "archive_members"
    repeat = extra / "Repeat"
    selected = extra / "frozen"
    supplementary = extra / "supplementary"
    inputs = {
        "original": [data / row["relative"] for row in INPUT_MANIFEST],
        "publication": [archive / path for path in ARCHIVED_MEMBERS if not path.endswith((".h5", ".joblib"))],
        "permutation": [supplementary / f"Supplementary Material {i}.xlsx" for i in range(1, 4)],
        "rf_spatial": [data / f"data/PCA_with_labels_{part}.csv" for part in ("Train", "Test")]
        + [data / f"original_analyses/grayscale_data_combined_{part}.csv" for part in ("train", "test")],
        "repeat10": list(repeat_inputs(repeat)),
        "frozen42": list(frozen_inputs(selected))
        + [data / "PCA_test/PCA_test/PCA_variance_contributions.xlsx"],
        "importance": [archive / path for path in ARCHIVED_MEMBERS if path.endswith((".h5", ".joblib"))],
    }
    grids = [data / "data/PCA_with_labels_Test.csv",
             data / "original_analyses/grayscale_data_combined_test.csv"]
    for stage in ("frozen42", "fixed_spatial", "importance"):
        inputs.setdefault(stage, []).extend(grids)
    for stage in ("publication", "repeat10"):
        inputs[stage].append(data / "data/Sample Location_delet_orogenic_gold.xlsx")
    if "importance" in args.only and "fixed_spatial" not in args.only:
        parser.error("importance 需要同次运行的 fixed_spatial 折模型")
    missing = sorted({str(path) for stage in args.only for path in inputs[stage] if not path.is_file()})
    if missing:
        parser.exit(2, "缺少输入\n" + "\n".join(missing) + "\n")
    if args.check:
        for stage in stages:
            if stage in args.only:
                print(stage, len(inputs[stage]), "inputs available")
        return
    label = args.label or datetime.now().strftime("CAOB_Review_Full_%Y%m%d_%H%M%S_%f")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
        parser.error("--label 只能是一级目录名")
    output = project / "result" / label
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CAOB_PROJECT=str(project), CAOB_SPATIAL=str(output / "Fixed_Label_Spatial_CV"),
               CAOB_OUTPUT=str(output), MPLCONFIGDIR=str(output / "_mpl_config"),
               MPLBACKEND="Agg", PYTHONUTF8="1")
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "tensorflow", "keras", "scipy",
                 "matplotlib", "openpyxl", "joblib", "Pillow", "h5py", "seaborn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    (output / "environment.json").write_text(json.dumps({
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "packages": packages}, indent=2), encoding="utf8")
    commands = {
        "original": [(str(Path(__file__).resolve()), ["--_original"])],
        "publication": [
            ("PA_saved_six_models", ["--archive", archive, "--output", output/"Publication_Figures/PA"]),
            ("PAC_manuscript", ["--archive", archive, "--output", output/"Publication_Figures/ROC_audit"]),
            ("PAC_publication_four", ["--input", output/"Publication_Figures/ROC_audit",
                                      "--output", output/"Publication_Figures/four_metrics"]),
        ],
        "permutation": [("caob_permutation_test", ["--tables", supplementary,
                                                  "--output", output/"Permutation_Test"])],
        "rf_spatial": [("RF_spatial_block_CV", ["--output", output/"RF_Spatial_Block"])],
        "fixed_spatial": [("fixed_labels_spatial_cv", []), ("render_fixed_labels_figure7", [])],
        "importance": [("reviewer4_feature_importance", ["--cv", output/"Fixed_Label_Spatial_CV",
                        "--models", archive, "--out", output/"Feature_Importance"]),
                       ("plot_reviewer4_figure9", []),
                       ("reviewer4_original_method_folds", []),
                       ("plot_reviewer4_figure9", ["--original-method"])],
        "repeat10": [("refresh_repeat_outputs", ["--base", repeat, "--output", output/"Repeat_10"])],
        "frozen42": [("frozen_models", ["--root", project, "--base", selected,
                                       "--output", output/"New_Negative_Sample"])],
    }
    report = {}
    report_file = output / "stages.json"
    with tempfile.TemporaryDirectory(prefix="_scripts_", dir=output) as temporary:
        work = Path(temporary)
        for name, source in REVIEW_MODULES.items():
            (work / f"{name}.py").write_text(source, encoding="utf8")
        for stage in stages:
            if stage not in args.only:
                continue
            print(stage, output, flush=True)
            report[stage] = "running"
            report_file.write_text(json.dumps(report, indent=2), encoding="utf8")
            try:
                for name, options in commands[stage]:
                    script = Path(name) if stage == "original" else work / f"{name}.py"
                    current_env = dict(env, CAOB_OUTPUT=str(output / "Original_Analysis")) if stage == "original" else env
                    subprocess.run([sys.executable, str(script), *map(str, options)],
                                   cwd=output, env=current_env, check=True)
                if stage == "original":
                    organize_original_outputs(output / "Original_Analysis")
                report[stage] = "completed"
            except subprocess.CalledProcessError as error:
                report[stage] = f"failed (exit {error.returncode})"
            finally:
                report_file.write_text(json.dumps(report, indent=2), encoding="utf8")
            if report[stage] != "completed" and stage == "fixed_spatial" and "importance" in args.only:
                commands["importance"] = []
        if "importance" in report and not commands["importance"]:
            report["importance"] = "skipped (fixed_spatial failed)"
            report_file.write_text(json.dumps(report, indent=2), encoding="utf8")
    if any(status != "completed" for status in report.values()):
        parser.exit(1, f"部分阶段未完成，参见 {report_file}\n")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--_original"]:
        del sys.argv[1]
        main()
    else:
        review_main()

