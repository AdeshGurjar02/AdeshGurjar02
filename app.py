from flask import Flask, render_template, request
import pickle as pk
import sklearn
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

model1 = pk.load(open('model/dtc.pkl', 'rb'))
model2 = pk.load(open('model/knn.pkl', 'rb'))
model3 = pk.load(open('model/lr.pkl','rb'))
scaler = pk.load(open('model/scaling.pkl', 'rb'))

@application.route('/')
def welcome():
    return render_template('index.html')

@application.route('/Pbydt', methods=['GET', 'POST'])
def prediction1():
    if request.method == 'POST':
        protocol_type = int(request.form.get('protocol_type'))
        service = int(request.form.get('service'))
        flag = int(request.form.get('flag'))
        src_bytes = int(request.form.get('src_bytes'))
        dst_bytes = int(request.form.get('dst_bytes'))
        count = int(request.form.get('count'))
        same_srv_rate = float(request.form.get('same_srv_rate'))
        diff_srv_rate = float(request.form.get('diff_srv_rate'))
        dst_host_srv_count = int(request.form.get('dst_host_srv_count'))
        dst_host_same_srv_rate = float(request.form.get('dst_host_same_srv_rate'))

        new_data_scaled = scaler.transform([[protocol_type, service, flag, src_bytes, dst_bytes, count, same_srv_rate, diff_srv_rate, dst_host_srv_count, dst_host_same_srv_rate]])
        result = model1.predict(new_data_scaled)

        return render_template('predictionbydt.html', result=result[0])
    else:
        return render_template('predictionbydt.html')

@application.route('/Pbyknn', methods=['GET', 'POST'])
def prediction2():
    if request.method == 'POST':
        protocol_type = int(request.form.get('protocol_type'))
        service = int(request.form.get('service'))
        flag = int(request.form.get('flag'))
        src_bytes = int(request.form.get('src_bytes'))
        dst_bytes = int(request.form.get('dst_bytes'))
        count = int(request.form.get('count'))
        same_srv_rate = float(request.form.get('same_srv_rate'))
        diff_srv_rate = float(request.form.get('diff_srv_rate'))
        dst_host_srv_count = int(request.form.get('dst_host_srv_count'))
        dst_host_same_srv_rate = float(request.form.get('dst_host_same_srv_rate'))

        new_data_scaled = scaler.transform([[protocol_type, service, flag, src_bytes, dst_bytes, count, same_srv_rate, diff_srv_rate, dst_host_srv_count, dst_host_same_srv_rate]])
        result = model2.predict(new_data_scaled)

        return render_template('predictionbyknn.html', result=result[0])
    else:
        return render_template('predictionbyknn.html')
    
@application.route('/Pbylr', methods=['GET', 'POST'])
def prediction3():
    if request.method == 'POST':
        protocol_type = int(request.form.get('protocol_type'))
        service = int(request.form.get('service'))
        flag = int(request.form.get('flag'))
        src_bytes = int(request.form.get('src_bytes'))
        dst_bytes = int(request.form.get('dst_bytes'))
        count = int(request.form.get('count'))
        same_srv_rate = float(request.form.get('same_srv_rate'))
        diff_srv_rate = float(request.form.get('diff_srv_rate'))
        dst_host_srv_count = int(request.form.get('dst_host_srv_count'))
        dst_host_same_srv_rate = float(request.form.get('dst_host_same_srv_rate'))

        new_data_scaled = scaler.transform([[protocol_type, service, flag, src_bytes, dst_bytes, count, same_srv_rate, diff_srv_rate, dst_host_srv_count, dst_host_same_srv_rate]])
        result = model3.predict(new_data_scaled)

        return render_template('predictionbylr.html', result=result[0])
    else:
        return render_template('predictionbylr.html')


if __name__ == '__main__':
    application.run(debug=True)
