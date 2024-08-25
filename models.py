from transformers import AutoModelForAudioClassification
import torch
import os
import warnings
warnings.filterwarnings('ignore')

##########################################################################

def model_ft(label2id, id2label, num_classes=4, device="cpu"):
    MODEL_CKPT = "facebook/wav2vec2-base"

    model = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                            num_labels=num_classes,
                                                            label2id=label2id,
                                                            id2label= id2label
                                                            )
    model.freeze_feature_encoder()
    model.to(device)
    return model

##########################################################################

def model_kd(label2id, id2label, num_classes=4, device="cpu"):
    MODEL_CKPT = "facebook/wav2vec2-base"

    teacher = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                            num_labels=num_classes,
                                                            label2id=label2id,
                                                            id2label= id2label
                                                            )
    file_path_teacher = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/checkpoints/ftwav2vec2testsplit2_iemocap_multilingual.pth"
    print("START: Load Multilingual Teacher's Knowledge")
    if os.path.exists(file_path_teacher):
        checkpoint = torch.load(file_path_teacher)
        teacher.load_state_dict(checkpoint['model_state_dict'])
        print("Teacher model checkpoint has been loaded")
    print("END: Load Multilingual Teacher's Knowledge")
    teacher.freeze_feature_encoder()
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.to(device)

    student = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                            num_labels=num_classes,
                                                            label2id=label2id,
                                                            id2label= id2label
                                                            )
    student.freeze_feature_encoder()
    student.to(device)
    return teacher, student

##########################################################################

def model_mtkd(label2id, id2label, num_classes=4, device="cpu"):
    MODEL_CKPT = "facebook/wav2vec2-base"
    teacher_en = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                                num_labels=num_classes,
                                                                label2id=label2id,
                                                                id2label= id2label
                                                                )
    file_path_teacher_en = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/checkpoints/ftwav2vec2testsplit2.pth"
    print("START: Load Monolingual English Teacher's Knowledge")
    if os.path.exists(file_path_teacher_en):
        checkpoint = torch.load(file_path_teacher_en)
        teacher_en.load_state_dict(checkpoint['model_state_dict'])
        print("Teacher model checkpoint has been loaded")
    print("END: Load Monolingual English Teacher's Knowledge")
    teacher_en.freeze_feature_encoder()
    for param in teacher_en.parameters():
        param.requires_grad = False
    teacher_en.to(device)

    teacher_fi = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                                num_labels=num_classes,
                                                                label2id=label2id,
                                                                id2label= id2label
                                                                )
    file_path_teacher_fi = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/checkpoints_finnish/ftwav2vec2testsplit_JAKA.pth"
    print("START: Load Monolingual Finnish Teacher's Knowledge")
    if os.path.exists(file_path_teacher_fi):
        checkpoint = torch.load(file_path_teacher_fi)
        teacher_fi.load_state_dict(checkpoint['model_state_dict'])
        print("Teacher model checkpoint has been loaded")
    print("END: Load Monolingual Finnish Teacher's Knowledge")
    teacher_fi.freeze_feature_encoder()
    for param in teacher_fi.parameters():
        param.requires_grad = False
    teacher_fi.to(device)

    teacher_fr = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                                num_labels=num_classes,
                                                                label2id=label2id,
                                                                id2label= id2label
                                                                )
    file_path_teacher_fr = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/checkpoints_finnish/ftwav2vec2testsplit_CaFE.pth"
    print("START: Load Monolingual French Teacher's Knowledge")
    if os.path.exists(file_path_teacher_fr):
        checkpoint = torch.load(file_path_teacher_fr)
        teacher_fr.load_state_dict(checkpoint['model_state_dict'])
        print("Teacher model checkpoint has been loaded")
    print("END: Load Monolingual French Teacher's Knowledge")
    teacher_fr.freeze_feature_encoder()
    for param in teacher_fr.parameters():
        param.requires_grad = False
    teacher_fr.to(device)

    student = AutoModelForAudioClassification.from_pretrained(MODEL_CKPT,
                                                            num_labels=num_classes,
                                                            label2id=label2id,
                                                            id2label= id2label
                                                            )
    student.freeze_feature_encoder()
    student.to(device)

    return teacher_en, teacher_fi, teacher_fr, student

##########################################################################
