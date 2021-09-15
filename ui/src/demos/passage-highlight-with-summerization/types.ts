import { InvalidModelResponseError } from '@allenai/tugboat/error';
import { emory } from '@allenai/tugboat/lib';

export const Version = emory.getVersion('rc-v1');

export interface Input {
    passage: string;
    question: string;
}

export interface WithTokenizedInput {
    passage_tokens: string[];
    question_tokens: string[];
}

export const isWithTokenizedInput = (pred: any): pred is WithTokenizedInput => {
    const typedPred = pred as WithTokenizedInput;
    return Array.isArray(typedPred.passage_tokens) && Array.isArray(typedPred.question_tokens);
};

export interface TransformerQAPrediction {
    best_span_str: string[][];
    summarization: string[][];
    context: string[][];
    question: string[];
    answer: string[];
    tag: string[][];
}

export const isTransformerQAPrediction = (pred: Prediction): pred is TransformerQAPrediction => {
    const typedPred = pred as TransformerQAPrediction;
    return (
        // typedPred.best_span !== undefined &&
        // typedPred.best_span_scores !== undefined &&
        typedPred.best_span_str !== undefined
        // typedPred.context_tokens !== undefined &&
        // typedPred.id !== undefined &&
        // typedPred.span_end_logits !== undefined &&
        // typedPred.span_start_logits !== undefined
    );
};


export type Prediction =
    TransformerQAPrediction

export const isPrediction = (pred: Prediction): pred is Prediction => {
    const typedPred = pred as Prediction;
    return (
        isTransformerQAPrediction(typedPred)
    );
};

export const getBasicAnswer = (pred: Prediction): number | string | string[] => {
    const noAnswer = 'Answer not found.';
    if (isTransformerQAPrediction(pred)) {
        return pred.answer || noAnswer;
    }
    throw new InvalidModelResponseError(noAnswer);
};
