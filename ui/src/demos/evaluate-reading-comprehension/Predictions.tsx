import React from 'react';
import Tabs from 'antd/es/tabs';
import { Output } from '@allenai/tugboat/components';
import { Model } from '@allenai/tugboat/lib';
import { DebugInfo } from '../../components';
import {
    Input,
    Prediction,
    BiDAFPrediction,
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
    return <BasicPrediction input={input} output={output} />;
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
    output: BiDAFPrediction;
}) => {

    return (
        <>        
            <Tabs>
                {output.context.map((contexts, i) =>
                    <Tabs.TabPane tab={`${i}`} key={`${i}`}>
                        <Output.SubSection title="Passage" >
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
                        <BasicAnswer output={output.answer[i]} />
                    </Tabs.TabPane>
                )}
            </Tabs>
        </>
    );
};
