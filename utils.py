import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

##########################################################################

def update_iemocap_label(label, column="emo"):
    if label == "ang":
        return "anger"
    elif label == "hap":
        return "happiness"
    elif label == "neu":
        return "neutral"
    elif label == "sad":
        return "sadness"

##########################################################################

def update_fesc_label(label, column="emo"):
    if label == "1":
        return "anger"
    elif label == "2":
        return "happiness"
    elif label == "3":
        return "neutral"
    elif label == "4":
        return "sadness"

##########################################################################

def is_common(emotion):
    if emotion == "neutral" or emotion == "sadness" or emotion == "happiness" or emotion == "anger":
        return 1
    else:
        return 0

########################################################################## 

def update_iemocap_path(path):
    return "/m/triton/scratch/elec/t405-puhe/p/bijoym1/TempDatasets" + path.split("/m/teamwork/t40511_asr/c")[-1]

##########################################################################

# def update_cafe_path(path):
#     return "/scratch/elec/t405-puhe/p/bijoym1/TempDatasets" + path.split("/m/teamwork/t40511_asr/c/CaFE/data")[-1]

##########################################################################

def update_cafe_path(path):
    return "/content" + path

##########################################################################

# this fn is for Finnish dataset only
def update_file_path(path):
    return "/m/triton/scratch/elec/" + path.split("/m/triton/scratch/")[1]

##########################################################################

def update_audio_path(path):
    return "/m/triton/scratch/elec/t405-puhe/p/bijoym1/TempDatasets" + path.split("/m/teamwork/t40511_asr/c")[-1]

##########################################################################

# this fn is for CaFE dataset only
def update_wav_path_cafe(path):
    return "/m/triton/scratch/elec/t405-puhe/p/bijoym1/TempDatasets" + path.split("/m/teamwork/t40511_asr/c/CaFE/data")[-1]

##########################################################################

def to_label(emo, label2id):
    return label2id[emo]

##########################################################################

def preprocess_function(examples, feature_extractor):
    '''
    This function prepares the dataset for the transformer
    by applying the feature extractor to it (among other
    processes).
    '''
    max_duration = 20.0 # 20.0 # seconds
    # max_duration = 120.0 # seconds
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(audio_arrays,
                               sampling_rate=feature_extractor.sampling_rate,
                               max_length=int(feature_extractor.sampling_rate * max_duration),
                               padding=True,
                               truncation=True)
    return inputs

##########################################################################

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x["input_values"]), reverse=True)
    inputs = [torch.tensor(example["input_values"]) for example in batch]
    labels = [int(example["label"]) for example in batch]
    inputs = pad_sequence(inputs, batch_first=True)
    return {"input_values": inputs, "label": torch.tensor(labels)}

##########################################################################

def train(model, train_loader, optimizer, loss_fn, device):
    model.train()

    total_predictions = 0
    correct_predictions = 0
    total_train_loss = 0

    # train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_OF_EPOCHS}, Training", leave=False)

    all_actual_labels = []
    all_predictions = []
    train_confidence_scores = []

    for batch in train_loader:
        inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        outputs = outputs.logits

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        train_loader.set_postfix(loss=loss.item())

        _, predicted = torch.max(outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        total_train_loss += loss.item()

        all_actual_labels.extend(labels.tolist())
        all_predictions.extend(predicted.tolist())

        max_values, max_indices = torch.softmax(outputs, dim=1).max(dim=1)
        probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
        train_confidence_scores.extend(probability_label_tuples)

    total_train_loss /= len(train_loader)
    train_accuracy = correct_predictions / total_predictions

    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    return (unweighted_recall, weighted_recall, train_accuracy, total_train_loss, train_confidence_scores)

##########################################################################

def train_kd(teacher, student, train_loader, optimizer_ce, optimizer_kl, loss_fn_ce, loss_fn_kl, device, temperature=5, lambda_param=0.25):
    teacher.train()
    student.train()

    total_predictions = 0
    correct_predictions = 0
    total_train_loss = 0
    total_kl_loss = 0
    total_ce_loss = 0

    # train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_OF_EPOCHS}, Training", leave=False)

    all_actual_labels = []
    all_predictions = []
    train_confidence_scores = []

    for batch in train_loader:
        inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

        optimizer_ce.zero_grad()
        optimizer_kl.zero_grad()

        teacher_outputs = teacher(inputs)
        teacher_outputs = teacher_outputs.logits
        teacher_outputs_soft = F.softmax(teacher_outputs / temperature, dim=-1)

        student_outputs = student(inputs)
        student_outputs = student_outputs.logits
        # student_outputs_soft = F.softmax(student_outputs / temperature, dim=-1)
        student_outputs_soft = F.log_softmax(student_outputs / temperature, dim=-1)

        loss_kl = loss_fn_kl(student_outputs_soft, teacher_outputs_soft) * (temperature**2)
        loss_ce = loss_fn_ce(student_outputs, labels)

        loss = (1. - lambda_param) * loss_ce + lambda_param * loss_kl

        loss.backward()

        optimizer_ce.step()

        train_loader.set_postfix(loss=loss.item())

        _, predicted = torch.max(student_outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        total_train_loss += loss.item()
        total_kl_loss += loss_kl.item()
        total_ce_loss += loss_ce.item()

        all_actual_labels.extend(labels.tolist())
        all_predictions.extend(predicted.tolist())

        max_values, max_indices = torch.softmax(student_outputs, dim=1).max(dim=1)
        probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
        train_confidence_scores.extend(probability_label_tuples)

    total_train_loss /= len(train_loader)
    total_kl_loss /= len(train_loader)
    total_ce_loss /= len(train_loader)
    train_accuracy = correct_predictions / total_predictions

    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    return (unweighted_recall, weighted_recall, train_accuracy, total_train_loss, total_kl_loss, total_ce_loss, train_confidence_scores)

##########################################################################

def calculate_cosine_similarity(tensor1, tensor2):
    tensor1_flat = tensor1.view(tensor1.size(0), -1)
    tensor2_flat = tensor2.view(tensor2.size(0), -1)
    cosine_sim = F.cosine_similarity(tensor1_flat, tensor2_flat, dim=1)
    return cosine_sim.mean()

##########################################################################

def train_mtkd(teacher_en, teacher_fi, teacher_fr, student, train_loader, optimizer_ce, optimizer_kl, loss_fn_ce, loss_fn_kl, device, temperature=5, lambda_param=0.25):
    teacher_en.train()
    teacher_fi.train()
    teacher_fr.train()
    student.train()

    total_predictions = 0
    correct_predictions = 0
    total_train_loss = 0
    total_kl_loss = 0
    total_ce_loss = 0

    all_actual_labels = []
    all_predictions = []
    train_confidence_scores = []

    for batch in train_loader:
        inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

        optimizer_ce.zero_grad()
        optimizer_kl.zero_grad()

        teacher_en_outputs = teacher_en(inputs)
        teacher_en_outputs = teacher_en_outputs.logits
        teacher_en_outputs_soft = F.softmax(teacher_en_outputs / temperature, dim=-1)

        teacher_fi_outputs = teacher_fi(inputs)
        teacher_fi_outputs = teacher_fi_outputs.logits
        teacher_fi_outputs_soft = F.softmax(teacher_fi_outputs / temperature, dim=-1)

        teacher_fr_outputs = teacher_fr(inputs)
        teacher_fr_outputs = teacher_fr_outputs.logits
        teacher_fr_outputs_soft = F.softmax(teacher_fr_outputs / temperature, dim=-1)

        student_outputs = student(inputs)
        student_outputs = student_outputs.logits
        student_outputs_soft = F.log_softmax(student_outputs / temperature, dim=-1)
        # student_outputs_soft = F.softmax(student_outputs / temperature, dim=-1)

        cos_sim_en = calculate_cosine_similarity(student_outputs, teacher_en_outputs)
        cos_sim_fi = calculate_cosine_similarity(student_outputs, teacher_fi_outputs)
        cos_sim_fr = calculate_cosine_similarity(student_outputs, teacher_fr_outputs)

        cosine_sims = torch.tensor([cos_sim_en, cos_sim_fi, cos_sim_fr])
        weights = F.softmax(cosine_sims / 0.25, dim=-1)
        weight_en, weight_fi, weight_fr = weights

        print(f"Cosine Similarity: EN = {cos_sim_en:.4}, FI = {cos_sim_fi:.4}, FR = {cos_sim_fr:.4}")
        print(f"Softmax HighTemp.: EN = {weight_en:.4}, FI = {weight_fi:.4}, FR = {weight_fr:.4}")

        loss_teacher_en = loss_fn_kl(student_outputs_soft, teacher_en_outputs_soft) * (temperature**2)
        loss_teacher_fi = loss_fn_kl(student_outputs_soft, teacher_fi_outputs_soft) * (temperature**2)
        loss_teacher_fr = loss_fn_kl(student_outputs_soft, teacher_fr_outputs_soft) * (temperature**2)

        loss_ce = loss_fn_ce(student_outputs, labels)

        # currently on iemocap => English
        # loss_kl = 0.33*loss_teacher_en + 0.33*loss_teacher_fi + 0.33*loss_teacher_fr
        # loss_kl = 0.1*loss_teacher_en + 0.8*loss_teacher_fi + 0.1*loss_teacher_fr
        loss_kl = weight_en * loss_teacher_en + weight_fi * loss_teacher_fi + weight_fr * loss_teacher_fr


        loss = (1. - lambda_param) * loss_ce + lambda_param * loss_kl

        loss.backward()

        optimizer_ce.step()

        train_loader.set_postfix(loss=loss.item())

        _, predicted = torch.max(student_outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        total_train_loss += loss.item()
        total_kl_loss += loss_kl.item()
        total_ce_loss += loss_ce.item()

        all_actual_labels.extend(labels.tolist())
        all_predictions.extend(predicted.tolist())

        max_values, max_indices = torch.softmax(student_outputs, dim=1).max(dim=1)
        probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
        train_confidence_scores.extend(probability_label_tuples)

    total_train_loss /= len(train_loader)
    total_kl_loss /= len(train_loader)
    total_ce_loss /= len(train_loader)
    train_accuracy = correct_predictions / total_predictions

    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    return (unweighted_recall, weighted_recall, train_accuracy, total_train_loss, total_kl_loss, total_ce_loss, train_confidence_scores)

##########################################################################

def validation_v2(model, valid_loader, loss_fn, device):
    model.eval()

    total_predictions = 0
    correct_predictions = 0
    total_valid_loss = 0

    all_actual_labels = []
    all_predictions = []
    valid_confidence_scores = []

    with torch.no_grad():
        for batch in valid_loader:
            inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

            outputs = model(inputs)
            outputs = outputs.logits

            loss = loss_fn(outputs, labels)

            valid_loader.set_postfix(loss=loss.item())

            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            total_valid_loss += loss.item()

            all_actual_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

            max_values, max_indices = torch.softmax(outputs, dim=1).max(dim=1)
            probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
            valid_confidence_scores.extend(probability_label_tuples)

        total_valid_loss /= len(valid_loader)
        valid_accuracy = correct_predictions / total_predictions
    
    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    return (unweighted_recall, weighted_recall, valid_accuracy, total_valid_loss, valid_confidence_scores, all_actual_labels, all_predictions)

##########################################################################

def validation(model, valid_loader, loss_fn, device):
    model.eval()

    total_predictions = 0
    correct_predictions = 0
    total_valid_loss = 0

    all_actual_labels = []
    all_predictions = []
    valid_confidence_scores = []

    with torch.no_grad():
        for batch in valid_loader:
            inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

            outputs = model(inputs)
            outputs = outputs.logits

            loss = loss_fn(outputs, labels)

            valid_loader.set_postfix(loss=loss.item())

            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            total_valid_loss += loss.item()

            all_actual_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

            max_values, max_indices = torch.softmax(outputs, dim=1).max(dim=1)
            probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
            valid_confidence_scores.extend(probability_label_tuples)

        total_valid_loss /= len(valid_loader)
        valid_accuracy = correct_predictions / total_predictions
    
    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    return (unweighted_recall, weighted_recall, valid_accuracy, total_valid_loss, valid_confidence_scores, all_actual_labels, all_predictions)

##########################################################################

def validation_kd(teacher, student, valid_loader, loss_fn_ce, loss_fn_kl, device, temperature=5, lambda_param=0.25):
    teacher.eval()
    student.eval()

    total_predictions = 0
    correct_predictions = 0
    total_valid_loss = 0
    total_kl_loss = 0
    total_ce_loss = 0

    all_actual_labels = []
    all_predictions = []
    valid_confidence_scores = []

    with torch.no_grad():
        for batch in valid_loader:
            inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

            teacher_outputs = teacher(inputs)
            teacher_outputs = teacher_outputs.logits
            teacher_outputs_soft = F.softmax(teacher_outputs / temperature, dim=-1)

            student_outputs = student(inputs)
            student_outputs = student_outputs.logits
            # student_outputs_soft = F.softmax(student_outputs / temperature, dim=-1)
            student_outputs_soft = F.log_softmax(student_outputs / temperature, dim=-1)

            loss_kl = loss_fn_kl(student_outputs_soft, teacher_outputs_soft) * (temperature**2)
            loss_ce = loss_fn_ce(student_outputs, labels)

            loss = (1. - lambda_param) * loss_ce + lambda_param * loss_kl


            valid_loader.set_postfix(loss=loss.item())

            _, predicted = torch.max(student_outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            total_valid_loss += loss.item()
            total_kl_loss += loss_kl.item()
            total_ce_loss += loss_ce.item()

            all_actual_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

            max_values, max_indices = torch.softmax(student_outputs, dim=1).max(dim=1)
            probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
            valid_confidence_scores.extend(probability_label_tuples)

        total_valid_loss /= len(valid_loader)
        total_kl_loss /= len(valid_loader)
        total_ce_loss /= len(valid_loader)
        valid_accuracy = correct_predictions / total_predictions
    
    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    return (unweighted_recall, weighted_recall, valid_accuracy, total_valid_loss, total_kl_loss, total_ce_loss, valid_confidence_scores, all_actual_labels, all_predictions)

##########################################################################

def validation_mtkd(teacher_en, teacher_fi, teacher_fr, student, valid_loader, loss_fn_ce, loss_fn_kl, device, temperature=5, lambda_param=0.25):
    teacher_en.eval()
    teacher_fi.eval()
    teacher_fr.eval()
    student.eval()

    total_predictions = 0
    correct_predictions = 0
    total_valid_loss = 0
    total_kl_loss = 0
    total_ce_loss = 0

    all_actual_labels = []
    all_predictions = []
    valid_confidence_scores = []

    with torch.no_grad():
        for batch in valid_loader:
            inputs, labels = batch['input_values'].to(device), batch['label'].to(device)

            teacher_en_outputs = teacher_en(inputs)
            teacher_en_outputs = teacher_en_outputs.logits
            teacher_en_outputs_soft = F.softmax(teacher_en_outputs / temperature, dim=-1)

            teacher_fi_outputs = teacher_fi(inputs)
            teacher_fi_outputs = teacher_fi_outputs.logits
            teacher_fi_outputs_soft = F.softmax(teacher_fi_outputs / temperature, dim=-1)

            teacher_fr_outputs = teacher_fr(inputs)
            teacher_fr_outputs = teacher_fr_outputs.logits
            teacher_fr_outputs_soft = F.softmax(teacher_fr_outputs / temperature, dim=-1)

            student_outputs = student(inputs)
            student_outputs = student_outputs.logits
            student_outputs_soft = F.log_softmax(student_outputs / temperature, dim=-1)
            
            cos_sim_en = calculate_cosine_similarity(student_outputs, teacher_en_outputs)
            cos_sim_fi = calculate_cosine_similarity(student_outputs, teacher_fi_outputs)
            cos_sim_fr = calculate_cosine_similarity(student_outputs, teacher_fr_outputs)

            cosine_sims = torch.tensor([cos_sim_en, cos_sim_fi, cos_sim_fr])
            weights = F.softmax(cosine_sims / 0.25, dim=-1)
            weight_en, weight_fi, weight_fr = weights

            print(f"Cosine Similarity: EN = {cos_sim_en:.4}, FI = {cos_sim_fi:.4}, FR = {cos_sim_fr:.4}")
            print(f"Softmax HighTemp.: EN = {weight_en:.4}, FI = {weight_fi:.4}, FR = {weight_fr:.4}")


            loss_teacher_en = loss_fn_kl(student_outputs_soft, teacher_en_outputs_soft) * (temperature**2)
            loss_teacher_fi = loss_fn_kl(student_outputs_soft, teacher_fi_outputs_soft) * (temperature**2)
            loss_teacher_fr = loss_fn_kl(student_outputs_soft, teacher_fr_outputs_soft) * (temperature**2)

            loss_ce = loss_fn_ce(student_outputs, labels)

            # currently on iemocap => English
            # loss_kl = 0.33*loss_teacher_en + 0.33*loss_teacher_fi + 0.33*loss_teacher_fr
            # loss_kl = 0.1*loss_teacher_en + 0.8*loss_teacher_fi + 0.1*loss_teacher_fr
            loss_kl = weight_en * loss_teacher_en + weight_fi * loss_teacher_fi + weight_fr * loss_teacher_fr

            loss = (1. - lambda_param) * loss_ce + lambda_param * loss_kl

            valid_loader.set_postfix(loss=loss.item())

            _, predicted = torch.max(student_outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            total_valid_loss += loss.item()
            total_kl_loss += loss_kl.item()
            total_ce_loss += loss_ce.item()

            all_actual_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

            max_values, max_indices = torch.softmax(student_outputs, dim=1).max(dim=1)
            probability_label_tuples = [(max_value.item(), max_index.item()) for max_value, max_index in zip(max_values, max_indices)]
            valid_confidence_scores.extend(probability_label_tuples)

        total_valid_loss /= len(valid_loader)
        total_kl_loss /= len(valid_loader)
        total_ce_loss /= len(valid_loader)
        valid_accuracy = correct_predictions / total_predictions
    
    unweighted_recall = recall_score(all_actual_labels, all_predictions, average='macro')
    weighted_recall = recall_score(all_actual_labels, all_predictions, average='weighted')

    return (unweighted_recall, weighted_recall, valid_accuracy, total_valid_loss, total_kl_loss, total_ce_loss, valid_confidence_scores, all_actual_labels, all_predictions)

##########################################################################

def keep_first_n(data, n=10):
    class_count = {}
    filtered_data = []

    for item in data:
        label = item[1]
        
        if label not in class_count:
            class_count[label] = 0

        if class_count[label] < n:
            filtered_data.append(item)
            class_count[label] += 1

    return filtered_data

##########################################################################

def plot_confidence_scores(data, title, file_id):
    data = sorted(data, key=lambda x: x[1])
    data = keep_first_n(data, n=12)
    probabilities, class_labels = zip(*data)
    plt.figure(figsize=(12, 5))
    plt.bar([x for x in range(len(class_labels))], probabilities, color='skyblue')
    plt.xlabel('Class Label')
    plt.ylabel('Probability Score')
    plt.title(title)
    plt.xticks(range(len(data)), class_labels)
    filename = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/SER/FTWav2Vec2/figures/cs_{title.lower().split()[0]}_{file_id}.jpg"
    plt.savefig(filename, format='jpg')
    plt.show()

##########################################################################

def plot_confusion_matrix(actual, predicted, file_id):
    data = confusion_matrix(actual, predicted)
    num_classes = max(max(actual), max(predicted)) + 1  # Get the maximum class label present in actual and predicted lists
    df_cm = pd.DataFrame(data, range(num_classes), range(num_classes))

    plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="Blues", annot_kws={"size": 12})

    plt.ylabel("Actual")
    plt.xlabel("Predicted")  # Corrected xlabel
    plt.title(f"Session {file_id}")

    filename = f"cm_{file_id}.jpg"
    plt.savefig(filename, format='jpg')
    plt.show()

##########################################################################
