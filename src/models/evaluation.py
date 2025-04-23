import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
import os
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def evaluate_multivariate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    data_info: Dict[str, Any],
    plot: bool = True,
    save_plot: bool = True,
    plot_dir: str = "plots"
) -> Dict[str, Any]:
    """
    다변량 시계열 분류 모델을 평가합니다.
    
    Args:
        model (torch.nn.Module): 평가할 모델
        test_loader (DataLoader): 테스트 데이터 로더
        data_info (Dict[str, Any]): 데이터 정보
        plot (bool): 결과 시각화 여부
        save_plot (bool): 시각화 결과 저장 여부
        plot_dir (str): 시각화 결과 저장 디렉토리
    
    Returns:
        Dict[str, Any]: 평가 지표 및 결과
    """
    device = torch.device(data_info['device'])
    model.eval()
    
    # 결과 저장을 위한 리스트
    all_preds = []
    all_targets = []
    
    # 클래스 매핑 정보
    inverse_class_mapping = data_info['inverse_class_mapping']
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # CPU로 이동하여 NumPy 배열로 변환
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # NumPy 배열로 변환
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 분류 지표 계산
    accuracy = accuracy_score(all_targets, all_preds)
    
    # 클래스별 정밀도, 재현율, F1 점수
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None
    )
    
    # 혼동 행렬 계산
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # 클래스 레이블 변환 (숫자 -> 원래 레이블)
    class_labels = [inverse_class_mapping[i] for i in range(len(inverse_class_mapping))]
    
    # 분류 보고서
    clf_report = classification_report(
        all_targets, all_preds, 
        target_names=class_labels, 
        output_dict=True
    )
    
    logger.info(f"테스트 정확도: {accuracy:.4f}")
    logger.info(f"클래스별 F1 점수: {f1}")
    
    # 결과 시각화
    if plot:
        # 결과 저장 디렉토리 생성
        if save_plot:
            os.makedirs(plot_dir, exist_ok=True)
        
        # 혼동 행렬 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title('혼동 행렬')
        plt.ylabel('실제 클래스')
        plt.xlabel('예측 클래스')
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
            logger.info(f"혼동 행렬 저장: {os.path.join(plot_dir, 'confusion_matrix.png')}")
        
        if not save_plot:
            plt.show()
        plt.close()
        
        # 클래스별 성능 지표 시각화
        plt.figure(figsize=(12, 6))
        x = np.arange(len(class_labels))
        width = 0.2
        
        plt.bar(x - width, precision, width=width, label='Precision')
        plt.bar(x, recall, width=width, label='Recall')
        plt.bar(x + width, f1, width=width, label='F1 Score')
        
        plt.xlabel('클래스')
        plt.ylabel('점수')
        plt.title('클래스별 성능 지표')
        plt.xticks(x, class_labels, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(plot_dir, 'class_metrics.png'))
            logger.info(f"클래스별 성능 지표 저장: {os.path.join(plot_dir, 'class_metrics.png')}")
        
        if not save_plot:
            plt.show()
        plt.close()
    
    # 평가 결과 반환
    evaluation_result = {
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1.tolist(),
        'support': support.tolist(),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': clf_report,
        'class_labels': class_labels
    }
    
    return evaluation_result


def analyze_misclassifications(
    model: torch.nn.Module,
    test_loader: DataLoader,
    data_info: Dict[str, Any],
    max_samples: int = 5,
    plot_dir: str = "plots"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    오분류된 샘플을 분석합니다.
    
    Args:
        model (torch.nn.Module): 평가할 모델
        test_loader (DataLoader): 테스트 데이터 로더
        data_info (Dict[str, Any]): 데이터 정보
        max_samples (int): 각 클래스별로 분석할 최대 샘플 수
        plot_dir (str): 시각화 결과 저장 디렉토리
    
    Returns:
        Dict[str, List[Dict[str, Any]]]: 오분류 분석 결과
    """
    device = torch.device(data_info['device'])
    model.eval()
    
    # 결과 저장을 위한 구조
    misclassified = defaultdict(list)
    
    # 클래스 매핑 정보
    inverse_class_mapping = data_info['inverse_class_mapping']
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # 오분류 샘플 식별
            mask = predicted != targets
            for i in range(len(mask)):
                if mask[i]:
                    true_class = targets[i].item()
                    pred_class = predicted[i].item()
                    true_label = inverse_class_mapping[true_class]
                    pred_label = inverse_class_mapping[pred_class]
                    
                    # 오분류 정보 저장
                    key = f"{true_label}_as_{pred_label}"
                    
                    # 각 클래스별 최대 샘플 수 제한
                    if len(misclassified[key]) < max_samples:
                        # 샘플 데이터 저장
                        sample_data = {
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'true_class': true_class,
                            'pred_class': pred_class,
                            'true_label': true_label,
                            'pred_label': pred_label,
                            'sequence': inputs[i].cpu().numpy(),
                            'confidence': torch.softmax(outputs[i], dim=0)[pred_class].item()
                        }
                        misclassified[key].append(sample_data)
    
    # 오분류 샘플 시각화
    os.makedirs(plot_dir, exist_ok=True)
    
    # 각 클래스별 오분류 분석
    for key, samples in misclassified.items():
        true_label, pred_label = key.split('_as_')
        
        # 샘플별 특성 시각화
        for idx, sample in enumerate(samples):
            plt.figure(figsize=(15, 5))
            sequence = sample['sequence']
            
            # 시퀀스 데이터의 각 특성별 시각화
            for feature_idx in range(min(5, sequence.shape[1])):  # 첫 5개 특성만 표시
                plt.plot(sequence[:, feature_idx], label=f'Feature {feature_idx}')
            
            plt.title(f"오분류 샘플 - 실제: {true_label}, 예측: {pred_label} (신뢰도: {sample['confidence']:.4f})")
            plt.xlabel('시퀀스 인덱스')
            plt.ylabel('특성 값')
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(plot_dir, f'misclassified_{true_label}_as_{pred_label}_{idx}.png'))
            plt.close()
    
    logger.info(f"오분류 분석 완료. {len(misclassified)} 클래스 조합에서 샘플 발견")
    
    return dict(misclassified)


def feature_importance_analysis(
    model: torch.nn.Module,
    test_loader: DataLoader,
    data_info: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
    plot: bool = True,
    plot_dir: str = "plots"
) -> Dict[str, List[float]]:
    """
    특성 중요도를 분석합니다.
    
    Args:
        model (torch.nn.Module): 평가할 모델
        test_loader (DataLoader): 테스트 데이터 로더
        data_info (Dict[str, Any]): 데이터 정보
        feature_names (List[str], optional): 특성 이름 리스트
        plot (bool): 결과 시각화 여부
        plot_dir (str): 시각화 결과 저장 디렉토리
    
    Returns:
        Dict[str, List[float]]: 특성 중요도 결과
    """
    device = torch.device(data_info['device'])
    model.eval()
    
    # 특성 수
    input_size = data_info['input_size']
    
    # 특성 이름 설정
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(input_size)]
    elif len(feature_names) < input_size:
        feature_names = feature_names + [f'Feature {i}' for i in range(len(feature_names), input_size)]
    
    # 순열 중요도 (Permutation Importance) 계산
    original_accuracy = 0
    feature_importance = np.zeros(input_size)
    
    # 원본 정확도 계산
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        original_accuracy = correct / total
    
    logger.info(f"원본 정확도: {original_accuracy:.4f}")
    
    # 각 특성별로 순열 중요도 계산
    for feature_idx in range(input_size):
        logger.info(f"특성 {feature_idx} ({feature_names[feature_idx]})의 중요도 분석 중...")
        
        permuted_accuracy = 0
        
        # 여러 순열 시도로 안정성 확보
        n_repeats = 3
        
        for _ in range(n_repeats):
            with torch.no_grad():
                correct = 0
                total = 0
                
                for inputs, targets in test_loader:
                    # 특성 값 순열화
                    permuted_inputs = inputs.clone()
                    perm_idx = torch.randperm(permuted_inputs.size(0))
                    permuted_inputs[:, :, feature_idx] = permuted_inputs[perm_idx, :, feature_idx]
                    
                    outputs = model(permuted_inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                
                # 이 반복에서의 정확도
                iter_accuracy = correct / total
                permuted_accuracy += iter_accuracy / n_repeats
        
        # 중요도 = 원본 정확도 - 순열 정확도
        feature_importance[feature_idx] = original_accuracy - permuted_accuracy
        logger.info(f"특성 {feature_idx} ({feature_names[feature_idx]}) 중요도: {feature_importance[feature_idx]:.4f}")
    
    # 중요도 정규화
    if np.sum(feature_importance) > 0:
        feature_importance = feature_importance / np.sum(feature_importance)
    
    # 중요도 시각화
    if plot:
        plt.figure(figsize=(12, 8))
        
        # 중요도 기준 정렬
        sorted_idx = np.argsort(feature_importance)
        sorted_feature_names = [feature_names[i] for i in sorted_idx]
        sorted_importance = feature_importance[sorted_idx]
        
        # 막대 그래프로 시각화
        plt.barh(range(len(sorted_feature_names)), sorted_importance, align='center')
        plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
        plt.xlabel('특성 중요도')
        plt.title('특성 중요도 분석 (순열 중요도)')
        plt.tight_layout()
        
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'feature_importance.png'))
        logger.info(f"특성 중요도 시각화 저장: {os.path.join(plot_dir, 'feature_importance.png')}")
        plt.close()
    
    # 결과 반환
    importance_result = {
        'feature_names': feature_names,
        'importance_scores': feature_importance.tolist()
    }
    
    return importance_result