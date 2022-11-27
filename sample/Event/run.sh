kappa_value=0.15
nevent="1000k"

python extract.py $kappa_value /home/r10222035/Boosted_V/sample/VBF_H5pp_ww_jjjj-c/Events/run_02/tag_1_delphes_events.root &
python extract.py $kappa_value /home/r10222035/Boosted_V/sample/VBF_H5mm_ww_jjjj-c/Events/run_02/tag_1_delphes_events.root &
python extract.py $kappa_value /home/r10222035/Boosted_V/sample/VBF_H5z_zz_jjjj-c/Events/run_02/tag_1_delphes_events.root &
python extract.py $kappa_value /home/r10222035/Boosted_V/sample/VBF_H5p_wz_jjjj/Events/run_02/tag_1_delphes_events.root &
python extract.py $kappa_value /home/r10222035/Boosted_V/sample/VBF_H5m_wz_jjjj/Events/run_02/tag_1_delphes_events.root &
python extract.py $kappa_value /home/r10222035/Boosted_V/sample/VBF_H5z_ww_jjjj/Events/run_02/tag_1_delphes_events.root &

wait

python convert.py /home/r10222035/Boosted_V/sample/event_samples_kappa$kappa_value-$nevent/VBF_H5pp_ww_jjjj.npy /home/r10222035/Boosted_V/sample/event_samples_kappa$kappa_value-$nevent/VBF_H5mm_ww_jjjj.npy /home/r10222035/Boosted_V/sample/event_samples_kappa$kappa_value-$nevent/VBF_H5z_zz_jjjj.npy /home/r10222035/Boosted_V/sample/event_samples_kappa$kappa_value-$nevent/VBF_H5p_wz_jjjj.npy /home/r10222035/Boosted_V/sample/event_samples_kappa$kappa_value-$nevent/VBF_H5m_wz_jjjj.npy /home/r10222035/Boosted_V/sample/event_samples_kappa$kappa_value-$nevent/VBF_H5z_ww_jjjj.npy &