import React from 'react';
import Tabs from 'antd/es/tabs';
import styled from 'styled-components';
import { 
    Output, 
    Highlight,
    HighlightColor,
    HighlightContainer,
    formatTokens,
    FormattedToken,
} from '@allenai/tugboat/components';
import { Model } from '@allenai/tugboat/lib';
import { DebugInfo } from '../../components';
import {
    Input,
    Prediction,
    TransformerQAPrediction,
} from './types';
interface Props {
    input: Input;
    model: Model;
    output: Prediction;
}
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
    return <BasicPrediction input={input} output={output} />;
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

  // Convert the tag label to a node type. In the long run this might make sense as
  // a map / lookup table of some sort -- but for now this works.
  let color: HighlightColor = 'T';
  let tooltip = " ";
  if (token.entity != 'O') {
    tooltip = summarization;
      color = 'T';
    }


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
    output: TransformerQAPrediction;
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