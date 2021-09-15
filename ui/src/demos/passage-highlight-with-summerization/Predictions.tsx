import React from 'react';
import Tabs from 'antd/es/tabs';
import styled from 'styled-components';
import { 
    TextWithHighlight, 
    Output, 
    ArithmeticEquation,
    Highlight,
    HighlightColor,
    HighlightContainer,
    formatTokens,
    FormattedToken,
} from '@allenai/tugboat/components';
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
// var Highlight = require('../../lib/highlighter')
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

const StyledHighlightContainer = styled(HighlightContainer)`
    margin-left: ${({ theme }) => theme.spacing.md};
`;

const StyledHighlight = styled(Highlight)`
    whiteSpace: 'pre-wrap'; 
    overflowWrap: 'break-word'; 
    wordWrap: 'break-word';
`

const TokenSpan = ({ token, id, summarization }: { token: FormattedToken; id: number; summarization: string}) => {
  if (token.entity === undefined) {
      // If no entity,
      // Display raw text.
      return <span>{`${token.text}${' '}`}</span>;
  }

  // const tagParts = token.entity.split('-');
  // const [tagLabel, attr] = tagParts;

  // Convert the tag label to a node type. In the long run this might make sense as
  // a map / lookup table of some sort -- but for now this works.
  let color: HighlightColor = 'T';
  let tooltip = " ";
  if (token.entity != 'O') {
    tooltip = summarization;
      color = 'T';
    }
  // } else if (tagLabel === 'a') {
  //     nodeType = 'Argument';
  //     color = 'M';
  // } else if (/ARG\d+/.test(tagLabel)) {
  //     nodeType = 'ASDFasdf';
  //     color = 'G';
  // } else if (tagLabel === 'R') {
  //     nodeType = 'Reference';
  //     color = 'P';
  // } else if (tagLabel === 'C') {
  //     nodeType = 'Continuation';
  //     color = 'A';
  // } else if (tagLabel === 'V') {
  //     nodeType = 'Verb';
  //     color = 'B';
  // }


  // Display entity text wrapped in a <Highlight /> component.
  return (
      <StyledHighlight id={id} label={token.entity} color={color} tooltip={tooltip}>
          {token.text}{' '}
      </StyledHighlight>
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
                        <Output.SubSection title="Passage" >
                            {contexts.map((c, j) => (
                                <div>
                                {formatTokens(output.tag[i][j], c).map((token, k) => (
                                        <TokenSpan key={k} id={k} token={token} summarization={output.summarization[i][j]} />
                                    ))}
                                </div>
                            ))}
                        </Output.SubSection>
                        <Output.SubSection title="Question">
                            <div>{output.question[0]}</div>
                        </Output.SubSection>
                        <BasicAnswer output={output.answer[i]} />
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
