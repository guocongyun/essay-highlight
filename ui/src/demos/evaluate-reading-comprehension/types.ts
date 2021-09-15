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

export interface BiDAFPrediction extends WithTokenizedInput {
    // best_span: number[]; // index into the token list
    best_span_str: string[][];
    context: string[][];
    question: string[];
    answer: string[];
    summarization: string[][];
    tag: string[][];
    // passage_question_attention: number[][];
    // span_end_logits: number[];
    // span_end_probs: number[];
    // span_start_logits: number[];
    // span_start_probs: number[];
    // token_offsets: number[][];
}

export const isBiDAFPrediction = (pred: Prediction): pred is BiDAFPrediction => {
    const typedPred = pred as BiDAFPrediction;
    return (
        // isWithTokenizedInput(pred) &&
        // typedPred.best_span !== undefined &&
        typedPred.best_span_str !== undefined 
        // typedPred.passage_question_attention !== undefined &&
        // typedPred.span_end_logits !== undefined &&
        // typedPred.span_end_probs !== undefined &&
        // typedPred.span_start_logits !== undefined &&
        // typedPred.span_start_probs !== undefined &&
        // typedPred.token_offsets !== undefined
    );
};

export type Prediction =
    | BiDAFPrediction

export const isPrediction = (pred: Prediction): pred is Prediction => {
    const typedPred = pred as Prediction;
    return (
        isBiDAFPrediction(typedPred)
    );
};

export const getBasicAnswer = (pred: Prediction): number | string | string[] => {
    const noAnswer = 'Answer not found.';
    if (isBiDAFPrediction(pred)) {
        return pred.answer || noAnswer;
    }
    throw new InvalidModelResponseError(noAnswer);
};
