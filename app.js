var net = require('net');
var k = "2";
var beacon_id = '0';
var Rssi = '0';
var exec = require('child_process').exec;
var arg1 = 'hello';
var arg2 = 'world';
var filename = 'Hashing_KNN.py'
var filename2 = '~/yolov7/realtimeD.py'


function full(Array) {
    for (let i = 0; i < Array.length; i++) {
        if (Array[i] == '0')
            return false;
    }
    return true;
}
let buffer = new Array(6);
buffer = ['0', '0', '0', '0', '0', '0'];
var Beacon_RSSi = '';
var t = 0;
var beacon_int = 0
var Rssi_int = 0;
var toclient = '0';
var rssi_result = '0'

var count = 0;
var fs = require('fs')

var clientHandler = function (socket) {



    //客戶端傳送資料的時候觸發data事件
    socket.on('data', function dataHandler(data) {//data是客戶端傳送給伺服器的資料

        //console.log(socket.remoteAddress, socket.remotePort, 'send', data.toString());
        k = data.toString();
        var K = k.split(",", 2);
        beacon_id = K[0];
        Rssi = K[1];

        let numStr = beacon_id.replace(/[^0-9]/ig, "");

        beacon_int = parseInt(numStr);
        Rssi_int = parseInt(Rssi);
        console.log(beacon_int, Rssi_int);
        Rssi_int = (Rssi_int + 101) / (-48 + 101);
        if (isNaN(Rssi_int)) {
            console.log(beacon_int);
            console.log("NAN");
        }
        if (beacon_int == 7) {
            buffer[5] = Rssi_int;
        }//因為用1 2 3 4 5 7
        else
            buffer[beacon_int - 1] = Rssi_int;

        if (full(buffer)) {
            Beacon_RSSi = `{q"Beacon_1q":q"${buffer[0]}q",q"Beacon_2q":q"${buffer[1]}q",q"Beacon_3q":q"${buffer[2]}q",q"Beacon_4q":q"${buffer[3]}q",q"Beacon_5q":q"${buffer[4]}q",q"Beacon_7q":q"${buffer[5]}q"}`;
            //console.log(Beacon_RSSi);
            exec('python' + ' ' + filename + ' ' + Beacon_RSSi + ' ' + count, function (err, stdout, stderr) {
                if (err) {
                    console.log('stderr', err);
                }
                if (stdout) {
                    console.log('output', stdout);
                    toclient = stdout;
                    rssi_result = stdout;
                    fs.writeFile('./rssi_result.txt', rssi_result, function (error) {
                        console.log(error)
                        console.log('文件寫入成功')
                    })
                    //socket.write(toclient);
                }
            });
            buffer = ['0', '0', '0', '0', '0', '0'];
            count++;
        }


        socket.write(toclient);
    });

    //當對方的連線斷開以後的事件
    socket.on('close', function () {
        //console.log(socket.remoteAddress, socket.remotePort, 'disconnected');
    })
    socket.on('error', (err) => {
        console.log(err);
    })
};

var app = net.createServer(clientHandler);

var exec = require('child_process').exec;
var child = exec('python' + ' ' + filename2 + ' ' + '--weights ~/yolov7/weights/yolov7.pt --source http://140.116.72.67:8080/yen --nosave');

child.stdout.on('data', function (data) {
    console.log('stdout: ' + data);
});

app.listen(8000, '140.116.72.77');//change to server
console.log('tcp server running on tcp://', '140.116.72.75', ':', 8000);



