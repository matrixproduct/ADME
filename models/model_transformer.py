from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import torch



import utils


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def transformer_predict(split, model_type='roberta', 
                        model_name='seyonec/ChemBERTa-zinc-base-v1', 
                        **kwargs):
    classification, num_classes =  utils.get_problem_type(split)

    if not num_classes:
        num_classes = 1
        
    loader = {key: df[['Drug', 'Y']] for key, df in split.items()}    
    

    # Setting optional model configuration
    model_args = ClassificationArgs()
    model_args.num_train_epochs = 1
    model_args.regression = not classification
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.overwrite_output_dir = True
    
    model_args.train_batch_size = 32

    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "mcc"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_patience = 2
    model_args.evaluate_during_training_steps = 4

    # Create a ClassificationModel
    model = ClassificationModel(
        model_type,
        model_name,
        num_labels=num_classes,
        args=model_args,
        use_cuda=utils.gpu_available,
    )

    # Train the model
    model.train_model(loader['train'])

    print('Training complete')

    # Evaluate the model
    result, valid_outputs, wrong_predictions = model.eval_model(loader['valid'])

    # Make predictions with the model
    X_test = list(loader['test']['Drug'])
    predictions, test_outputs = model.predict(X_test)
    if classification:
        torch_valid_logits = torch.from_numpy(valid_outputs)        
        valid_outputs = torch.nn.functional.softmax(torch_valid_logits, dim=1).detach().cpu().numpy()
        torch_test_logits = torch.from_numpy(test_outputs)        
        test_outputs = torch.nn.functional.softmax(torch_test_logits, dim=1).detach().cpu().numpy()
      
    
    return valid_outputs, test_outputs
    