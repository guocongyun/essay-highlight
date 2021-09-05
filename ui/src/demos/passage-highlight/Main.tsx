import React from 'react';
import Tabs from 'antd/es/tabs';
import {
    SelectedModelCard,
    Output,
    Field,
    Saliency,
    SelectExample,
    SelectModelAndDescription,
    Share,
    Submit,
    TaskDescription,
    TaskTitle,
} from '@allenai/tugboat/components';

import { AppId } from '../../AppId';
import { TaskDemo, Predict, Interpreters, Attackers } from '../../components';
import { config } from './config';
import { Usage } from './Usage';
import { Predictions } from './Predictions';
import { Input, Prediction, getBasicAnswer, isWithTokenizedInput, Version } from './types';
import { InterpreterData, DoubleGradInput, isDoubleInterpreterData } from '../../lib';
import 'react-dropzone-uploader/dist/styles.css'
import Dropzone from 'react-dropzone-uploader'
import PropTypes from 'prop-types';
import ReactPaginate from 'react-paginate';

const MyUploader = () => {
    // specify upload params and url for your files
    const getUploadParams = ({ meta }) => { return { url: 'http://localhost:8080/api/bidaf/upload' } }
    
    // called every time a file's `status` changes
    const handleChangeStatus = ({ meta, file }, status) => { console.log(status, meta, file) }
    
    // receives array of files that are done uploading when submit button is clicked
    const handleSubmit = (files) => { console.log(files.map(f => f.meta)) }
   
    return (
      <Dropzone
        getUploadParams={getUploadParams}
        onChangeStatus={handleChangeStatus}
        accept="*.zip"
      />
    )
}

const SingleFileAutoSubmit = () => {
    const toast = (innerHTML) => {
      const el = document.getElementById('toast')
      el.innerHTML = innerHTML
      el.className = 'show'
      setTimeout(() => { el.className = el.className.replace('show', '') }, 3000)
    }

    const getUploadParams = () => {
      return { url: 'http://localhost:8080/api/bidaf/upload' }
    }
  
    const handleChangeStatus = ({ meta, remove }, status) => {
      if (status === 'headers_received') {
        toast(`${meta.name} uploaded!`)
        remove()
      } else if (status === 'aborted') {
        toast(`${meta.name}, upload failed...`)
      }
    }
  
    return (
      <React.Fragment>
        <div id="toast">Upload</div>
        <Dropzone
          getUploadParams={getUploadParams}
          onChangeStatus={handleChangeStatus}
          maxFiles={1}
          multiple={false}
          inputContent="Drop A File"
        />
      </React.Fragment>
    )
  }

const Paginate = () => {
    return (
        <ReactPaginate
            previousLabel={'previous'}
            nextLabel={'next'}
            breakLabel={'...'}
            breakClassName={'break-me'}
            pageCount={10}
            marginPagesDisplayed={2}
            pageRangeDisplayed={5}
            // onPageChange={this.handlePageClick}
            containerClassName={'pagination'}
            activeClassName={'active'}
        />
    )
}

export const Main = () => {
    return (
        <TaskDemo ids={config.modelIds} taskId={config.taskId}>
            <TaskTitle />
            <TaskDescription />
            <SelectModelAndDescription />
            <SelectExample displayProp="question" placeholder="Select a Question…" />
            {/* <MyUploader/> */}
            <SingleFileAutoSubmit/>
            <Predict<Input, Prediction>
                version={Version}
                fields={
                    <>
                        <Field.Passage />
                        <Field.Question />
                        <Submit>Run Model</Submit>
                    </>
                }>
                {({ input, model, output }) => (
                    <Output>
                        <Output.Section
                            title="Model Output"
                            extra={
                                <Share.ShareButton
                                    doc={input}
                                    slug={Share.makeSlug(input.question)}
                                    type={Version}
                                    app={AppId}
                                />
                            }>
                            <Predictions input={input} model={model} output={output} />
                        </Output.Section>
                    </Output>
                )}
            </Predict>
        </TaskDemo>
    );
};
