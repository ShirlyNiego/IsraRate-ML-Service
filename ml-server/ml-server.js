// Input data should look like:
// {
//   tweets: [
//     {
//       id: "asdasd",
//       text: "i love israel"
//     },
//     {
//       id: "asdcascasv",
//       text: "bomb israel"
//     }
//   ]
// }

// Output data should look like:
// {
//   tweets: [
//     {
//       id: "asdasd",
//       tag: "1.0"
//     },
//     {
//       id: "asdcascasv",
//       text: "-1.0"
//     }
//   ]
// }

const { exec } = require('child_process');
const http = require('http');
const path = require('path')

var modelScript = path.join(__dirname,"model.py");

var processedTweets = 0;
var requiredTweets = 0;
outputTweets = [];
var minutes = 0.5;

var interval = minutes * 60 * 1000;
console.log("starting server!");

setInterval(function() {
  http.get('http://israrate-db.herokuapp.com/api/feed/GetRawFeedCount?limit=20', (resp) => {
    let data = '';

    // A chunk of data has been recieved.
    resp.on('data', (chunk) => {
      data += chunk;
    });

    // The whole response has been received. Print out the result.
    resp.on('end', () => {
      var respData = JSON.parse(data).data
      requiredTweets = respData.tweets.length;
      console.log("Starting to tag " + respData.tweets.length + " tweets");
      tagTweets(respData.tweets);
    });

  }).on("error", (err) => {
    console.log("Error: " + err.message);
  });
}, interval);

function tagTweets(inputTweetsArray) {
  // Go over all of the tweets
  for (var i = 0; i < inputTweetsArray.length; i++) {
      (function(i){
        var tweetText = inputTweetsArray[i].text.replace(/['"]+/g, '');
        
        // Run the model in new python child process
        exec('python -W ignore ' + modelScript + ' "' + tweetText + '"', (err, stdout, stderr) => {
          // Mark this tweet as processed
          processedTweets++;
          
          if (err) {
            console.error(`exec error: ${err}`);
            return;
          }

          // Get only the tag number
          stdout = stdout.substring(0, stdout.length - 2);

          outputTweets.push({
            id: inputTweetsArray[i].id,
            tag: stdout
          });

          //console.log("Tagged " + stdout);

          // Check if we have finished with all of the tweets we need to tag
          checkIfFinished();
        });
      })(i)               
  }
}

function checkIfFinished() {
  if (processedTweets >= requiredTweets) {
    processedTweets = 0;
    requiredTweets = 0;

    // Send the tagged tweets back to the DB
    sendTags({"tweets": outputTweets});
  }
}

function sendTags(outputTweetsArray) {

    let body = JSON.stringify(outputTweetsArray);

    const options = {
        hostname: 'israrate-db.herokuapp.com',
        port: 80,
        path: '/api/feed/setscore',
        method: 'POST',
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body)
      }
      };
    
    const req = http.request(options, (resp) => {
        console.log('Sent tagged tweets');
    });

    req.on('error', (e) => {
        console.error(`problem with request: ${e.message}`);
      });
    
    req.write(body);
    req.end();
}

// For testing
// tagTweets([
//   {
//     id: 111,
//     text: "bomb israel"
//   }, 
//   {
//     id: 222,
//     text: "i love israel"
//   }
// ]);
