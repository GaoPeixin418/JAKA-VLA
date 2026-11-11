"""Integration check for global Parquet indices, video offsets, RGB and identity."""
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from convert_zu7_for_openpi import convert, CAMERAS

with tempfile.TemporaryDirectory(prefix='zu7-conversion-',dir='/data/jaka-vla/results') as directory:
    root=Path(directory);source=root/'source';(source/'meta/episodes').mkdir(parents=True)
    (source/'data').mkdir();(source/'videos').mkdir()
    info={'codebase_version':'v3.0','fps':25,'total_episodes':4,'robot_type':'dual_jaka_zu7_lab_archive',
          'data_path':'data/{chunk_index}-{file_index}.parquet',
          'video_path':'videos/{video_key}-{chunk_index}-{file_index}.mp4',
          'features':{'observation.state':{'shape':[14]},'action':{'shape':[14]},
                      **{key:{'shape':[240,320,3]} for key in CAMERAS}}}
    (source/'meta/info.json').write_text(json.dumps(info))
    identity={'robot_id':'dual_jaka_zu7_lab_archive','test_identity':'must_preserve'}
    (source/'meta/project_identity.json').write_text(json.dumps(identity))
    episodes=[];expected={}
    for file_index in range(2):
        rows=[]
        for local_ep in range(2):
            ep=file_index*2+local_ep
            record={'episode_index':ep,'length':3,'dataset_from_index':ep*3,'data/chunk_index':0,
                    'data/file_index':file_index,'tasks':[f'pick target {ep}']}
            for key in CAMERAS:
                record.update({f'videos/{key}/chunk_index':0,f'videos/{key}/file_index':file_index,
                               f'videos/{key}/from_timestamp':local_ep*3/25})
            episodes.append(record)
            for frame in range(3):
                rows.append({'episode_index':ep,'frame_index':frame,'timestamp':frame/25,
                             'observation.state':[ep+frame/10]*14,'action':[ep+frame/10+.05]*14})
        pq.write_table(pa.Table.from_pylist(rows),source/f'data/0-{file_index}.parquet')
        for camera_index,key in enumerate(CAMERAS):
            path=source/f'videos/{key}-0-{file_index}.mp4'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25,(320,240))
            assert writer.isOpened()
            for i in range(6):
                img=np.zeros((240,320,3),np.uint8)
                img[:]=[25+file_index*70,30+i*20,40+camera_index*55]
                writer.write(img)
            writer.release()
            cap=cv2.VideoCapture(str(path))
            for i in range(6):
                ok,bgr=cap.read();assert ok
                expected[(file_index*2+i//3,i%3,CAMERAS[key])]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            cap.release()
    pq.write_table(pa.Table.from_pylist(episodes),source/'meta/episodes/part.parquet')
    output=root/'converted';manifest=convert(source,output,None)
    converted_info=json.loads((output/'meta/info.json').read_text())
    assert json.loads((output/'meta/project_identity.json').read_text())==identity
    assert converted_info['robot_type']==identity['robot_id']
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    dataset=LeRobotDataset(output.name,root=output)
    for index in range(12):
        row=dataset[index];ep,frame=divmod(index,3)
        np.testing.assert_allclose(row['state'],[ep+frame/10]*14,atol=1e-6)
        np.testing.assert_allclose(row['actions'],[ep+frame/10+.05]*14,atol=1e-6)
        for camera in CAMERAS.values():
            decoded=np.rint(row[camera].numpy().transpose(1,2,0)*255).astype(np.uint8)
            np.testing.assert_array_equal(decoded,expected[(ep,frame,camera)])
    print(json.dumps({'passed':True,'episodes':4,'parquet_files':2,'video_files':6,'frames':12,
                      'decoded_camera_frames_verified':36,'identity_preserved':True}),flush=True)
