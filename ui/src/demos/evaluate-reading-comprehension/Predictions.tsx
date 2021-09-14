import React from 'react';
import Tabs from 'antd/es/tabs';
import { TextWithHighlight, Output, ArithmeticEquation } from '@allenai/tugboat/components';
import { Model } from '@allenai/tugboat/lib';
import {
    InvalidModelResponseError,
    UnexpectedModelError,
    UnexpectedOutputError,
} from '@allenai/tugboat/error';
import { DebugInfo } from '../../components';
import { ModelId } from '../../lib';
import {
    Input,
    Prediction,
    BiDAFPrediction,
    TransformerQAPrediction,
    isBiDAFPrediction,
    isNAQANetPrediction,
    isTransformerQAPrediction,
    isNMNPrediction,
    getBasicAnswer,
} from './types';
interface Props {
    input: Input;
    model: Model;
    output: Prediction;
}
declare var require: any
var Highlight = require('react-highlighter')
export const Predictions = ({ input, model, output }: Props) => {
    return (
        <Output.Section>
            <OutputByModel input={input} output={output} model={model} />

            <DebugInfo input={input} output={output} model={model} />
        </Output.Section>
    );
};
const OutputByModel = ({
    input,
    output,
    model,
}: {
    input: Input;
    output: Prediction;
    model: Model;
}) => {
    switch (model.id) {
        case ModelId.Bidaf:
        case ModelId.BidafELMO:
        case ModelId.TransformerQA: {
            if (!isBiDAFPrediction(output) && !isTransformerQAPrediction(output)) {
                throw new UnexpectedOutputError(model.id);
            }
            return <BasicPrediction input={input} output={output} />;
        }
        case ModelId.Naqanet: {
            if (!isNAQANetPrediction(output)) {
                throw new UnexpectedOutputError(model.id);
            }
            return <NaqanetPrediction input={input} output={output} model={model} />;
        }
    }
    // If we dont have any output throw.
    throw new UnexpectedModelError(model.id);
};

const BasicAnswer = ({ output }: { output: any }) => {
    return (
        <Output.SubSection title="Answer">
            <div style={{whiteSpace: 'pre-wrap', overflowWrap: 'break-word', wordWrap: 'break-word'}}>{output}</div>
        </Output.SubSection>
    );
};

const BasicPrediction = ({
    input,
    output,
}: {
    input: Input;
    output: TransformerQAPrediction | BiDAFPrediction;
}) => {

    return (
        <>        
            <Tabs>
                {output.context.map((contexts, i) =>
                    <Tabs.TabPane tab={`${i}`} key={`${i}`}>
                        <BasicAnswer output={output.answer[i]} />
                        <Output.SubSection title="Passage Context" >
                            {contexts.map((context, j) => 
                                <Highlight 
                                    search={output.best_span_str[i][j]}
                                    matchElement={`font`}
                                    matchStyle={{background:"black", color:"white", fontSize:"10", flexShrink: 1, overflowWrap: 'break-word', wordWrap: 'break-word'}}>
                                    {output.context[i][j]}
                                </Highlight>
                            )}
                        </Output.SubSection>

                        <Output.SubSection title="Question">
                            <div>{output.question[0]}</div>
                        </Output.SubSection>
                    </Tabs.TabPane>
                )}
            </Tabs>
        </>
    );
};

// const NaqanetPrediction = ({
//     input,
//     output,
//     model,
// }: {
//     input: Input;
//     output: NAQANetPrediction;
//     model: Model;
// }) => {
//     // NAQANetAnswerType.PassageSpan
//     if (
//         isNAQANetPredictionSpan(output) &&
//         output.answer.answer_type === NAQANetAnswerType.PassageSpan
//     ) {
//         return (
//             <>
//                 <BasicAnswer output={output} />

//                 <Output.SubSection title="Explanation">
//                     The model decided the answer was in the passage.
//                 </Output.SubSection>

//                 <Output.SubSection title="Passage Context">
//                     <TextWithHighlight
//                         text={input.passage}
//                         highlights={output.answer.spans.map((s) => {
//                             return {
//                                 start: s[0],
//                                 end: s[1],
//                             };
//                         })}
//                     />
//                 </Output.SubSection>

//                 <Output.SubSection title="Question">
//                     <div>{input.question}</div>
//                 </Output.SubSection>
//             </>
//         );
//     }

//     // NAQANetAnswerType.QuestionSpan
//     if (
//         isNAQANetPredictionSpan(output) &&
//         output.answer.answer_type === NAQANetAnswerType.QuestionSpan
//     ) {
//         return (
//             <>
//                 <BasicAnswer output={output} />

//                 <Output.SubSection title="Explanation">
//                     The model decided the answer was in the question.
//                 </Output.SubSection>

//                 <Output.SubSection title="Passage Context">
//                     <div>{input.passage}</div>
//                 </Output.SubSection>

//                 <Output.SubSection title="Question">
//                     <TextWithHighlight
//                         text={input.question}
//                         highlights={output.answer.spans.map((s) => {
//                             return {
//                                 start: s[0],
//                                 end: s[1],
//                             };
//                         })}
//                     />
//                 </Output.SubSection>
//             </>
//         );
//     }

//     // NAQANetAnswerType.Count
//     if (isNAQANetPredictionCount(output)) {
//         return (
//             <>
//                 <BasicAnswer output={output} />

//                 <Output.SubSection title="Explanation">
//                     The model decided this was a counting problem.
//                 </Output.SubSection>

//                 <Output.SubSection title="Passage Context">
//                     <div>{input.passage}</div>
//                 </Output.SubSection>

//                 <Output.SubSection title="Question">
//                     <div>{input.question}</div>
//                 </Output.SubSection>
//             </>
//         );
//     }

//     // NAQANetAnswerType.Arithmetic
//     if (isNAQANetPredictionArithmetic(output)) {
//         // numbers include all numbers in the context, but we only care about ones that are positive or negative
//         const releventNumbers = (output.answer.numbers || []).filter((n) => n.sign !== 0);

//         return (
//             <>
//                 <BasicAnswer output={output} />

//                 <Output.SubSection title="Explanation">
//                     {releventNumbers.length ? (
//                         <div>
//                             The model used the arithmetic expression{' '}
//                             <ArithmeticEquation
//                                 numbersWithSign={releventNumbers}
//                                 answer={output.answer.value}
//                                 answerAtEnd={true}
//                             />
//                         </div>
//                     ) : (
//                         <div>The model decided this was an arithmetic problem.</div>
//                     )}
//                 </Output.SubSection>

//                 <Output.SubSection title="Passage Context">
//                     {releventNumbers.length ? (
//                         <TextWithHighlight
//                             text={input.passage}
//                             highlights={releventNumbers.map((n) => {
//                                 return {
//                                     start: n.span[0],
//                                     end: n.span[1],
//                                     color: n.sign > 0 ? 'G6' : 'R6',
//                                 };
//                             })}
//                         />
//                     ) : (
//                         <div>{input.passage}</div>
//                     )}
//                 </Output.SubSection>

//                 <Output.SubSection title="Question">
//                     <div>{input.question}</div>
//                 </Output.SubSection>
//             </>
//         );
//     }

//     // payload matched no known viz
//     throw new InvalidModelResponseError(model.id);
// };