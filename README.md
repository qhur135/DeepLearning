- 21.01.06 딥러닝 프로젝트 
<< Stanford Dog Dataset을 이용한 ResNet 모델의 레이어 수에 따른 성능 비교 >>

Stanford Dog Dataset
다운로드 링크 : http://vision.stanford.edu/aditya86/ImageNetDogs/

 A. 코드 간략하게 설명 (TransferLearning.py 내에 main함수 있음)
    1. ResNet 모델의 레이어 수에 따른 성능 비교
       (main함수 코드)
       model_ft = models.resnet18(pretrained=True) # resnet18 모델로 학습
       model_ft = models.resnet50(pretrained=True) # resnet50 모델로 학습
       model_ft = models.resnet152(pretrained=True) # resnet152 모델로 학습
       
    2. 전이학습 여부에 따른 성능 비교
       (main함수 코드)  
       model_ft = models.resnet18(pretrained=True) # 전이학습을 진행하는 경우
       model_ft = models.resnet18(pretrained=False) # 전이학습을 진행하지 않는 경우

    3. 매개변수 고정 여부에 따른 성능 비교
       (main함수 코드)
       for param in model_ft.parameters():  # 마지막 계층을 제외한 매개변수를 고정하는 경우
          param.requires_grad = False

