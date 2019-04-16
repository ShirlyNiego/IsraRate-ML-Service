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
var requiredTweets = 2;
outputTweets = [];

// TODO: Run in loop every X seconds
http.get('http://israrate-db.herokuapp.com/api/feed/GetRawFeedCount?limit=0', (resp) => {
  let data = '';

  // A chunk of data has been recieved.
  resp.on('data', (chunk) => {
    data += chunk;
  });

  // The whole response has been received. Print out the result.
  resp.on('end', () => {
    var respData = JSON.parse(data).data
    requiredTweets = respData.tweets.length;
    tagTweets(respData.tweets);
  });

}).on("error", (err) => {
  console.log("Error: " + err.message);
});


function tagTweets(inputTweetsArray) {
  // Go over all of the tweets
  for (var i = 0; i < inputTweetsArray.length; i++) {
      (function(i){
        // Run the model in new python child process
        exec('python -W ignore ' + modelScript + ' "' + inputTweetsArray[i].text + '"', (err, stdout, stderr) => {
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

          // Check if we have finished with all of the tweets we need to tag
          checkIfFinished();
        });
      })(i)               
  }
}

function checkIfFinished() {
  if (processedTweets >= requiredTweets) {
    console.log(outputTweets);
    processedTweets = 0;
    requiredTweets = 2;

    // Send the tagged tweets back to the DB
    sendTags(outputTweets);
  }
}

function sendTags(outputTweetsArray) {
    const options = {
        hostname: '',
        port: 3000,
        path: '/api/feed/setscore',
        method: 'POST'
      };
    
    const req = http.request(options, (resp) => {
        console.log('Sent tagged tweets');
    });

    req.on('error', (e) => {
        console.error(`problem with request: ${e.message}`);
      });
    
    req.write(JSON.stringify(outputTweetsArray));
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
