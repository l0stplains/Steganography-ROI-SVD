document.addEventListener('DOMContentLoaded', () => {
  // ----------------- EMBED SECTION -----------------
  const embedUploadForm = document.getElementById('embed-upload-form');
  const embedStartCamBtn = document.getElementById('embed-start-camera');
  const embedCamSection = document.getElementById('embed-camera-section');
  const embedCamStream = document.getElementById('embed-camera-stream');
  const embedCapturePhotoBtn = document.getElementById('embed-capture-photo');
  const embedDetectForm = document.getElementById('embed-detect-form');
  const embedDataForm = document.getElementById('embed-data-form');

  // Download Elements
  const downloadFormatSelect = document.getElementById('download-format');
  const downloadButton = document.getElementById('download-stego-button');

  // Stego Image Element
  const stImg = document.getElementById('embed-stego-image');

  // Variables to store filenames
  let stegoPngFilename = '';
  let stegoTiffFilename = '';

  // Camera helper
  function startCamera(videoElem) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => { videoElem.srcObject = stream; })
      .catch(err => {
        console.error("Camera error:", err);
        alert("Cannot access camera.");
      });
  }

  // EMBEDDING
  if(embedUploadForm){
    embedUploadForm.addEventListener('submit', e => {
      e.preventDefault();
      const formData = new FormData(embedUploadForm);
      formData.append('operation','embed');
      fetch('/upload',{
        method:'POST',
        body:formData
      })
      .then(r=>r.json())
      .then(d=>{
        if(d.error){
          alert(d.error);
        }
        else{
          const embedUploadedFilename = d.filename;
          const origImg = document.getElementById('embed-original-image');
          origImg.src = `/download/original/${embedUploadedFilename}`;
          origImg.onload=()=>{
            document.getElementById('embed-detection-section').style.display='block';
            alert("Image uploaded for embedding!");
          };
        }
      })
      .catch(er=>console.error("Embed upload error:",er));
    });
  }

  if(embedStartCamBtn){
    embedStartCamBtn.addEventListener('click', () => {
      embedCamSection.style.display='block';
      startCamera(embedCamStream);
    });
  }

  if(embedCapturePhotoBtn){
    embedCapturePhotoBtn.addEventListener('click', () => {
      const canvas = document.createElement('canvas');
      canvas.width = embedCamStream.videoWidth;
      canvas.height = embedCamStream.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(embedCamStream,0,0,canvas.width,canvas.height);

      canvas.toBlob(blob=>{
        const fData = new FormData();
        fData.append('image', blob, 'camera_photo.png'); // Save as PNG
        fData.append('operation','embed');
        fetch('/upload',{
          method:'POST',
          body:fData
        })
        .then(r=>r.json())
        .then(info=>{
          if(info.error){
            alert(info.error);
          }
          else{
            const embedUploadedFilename = info.filename;
            const embOrig = document.getElementById('embed-original-image');
            embOrig.src=`/download/original/${embedUploadedFilename}`;
            embOrig.onload=()=>{
              document.getElementById('embed-detection-section').style.display='block';
              alert("Camera photo uploaded for embedding!");
              embedCamStream.srcObject.getTracks().forEach(t=>t.stop());
              embedCamSection.style.display='none';
            };
          }
        })
        .catch(e=>console.error("Embed camera error:",e));
      }, 'image/png'); // Use PNG to prevent compression
    });
  }

  if(embedDetectForm){
    embedDetectForm.addEventListener('submit', e=>{
      e.preventDefault();
      const objClass = document.getElementById('embed-object-class').value;
      const embedUploadedFilename = getUploadedFilename();
      if(!embedUploadedFilename){
        alert("No image uploaded for embedding.");
        return;
      }
      fetch('/detect',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          filename: embedUploadedFilename,
          object_class: objClass,
          operation: 'embed'
        })
      })
      .then(r=>r.json())
      .then(dt=>{
        if(dt.error){
          alert(dt.error);
        }
        else{
          const embedDetectedFilename = dt.detected_filename;
          const embedMaxCapacity = dt.total_capacity_chars;
          document.getElementById('embed-max-capacity').innerText = embedMaxCapacity.toString();
          const detImg = document.getElementById('embed-detected-image');
          detImg.src = `/download/detected/${embedDetectedFilename}`;
          detImg.onload=()=>{
            document.getElementById('embed-results').style.display='block';
          };
        }
      })
      .catch(er=>console.error("Embed detect error:",er));
    });
  }

  if(embedDataForm){
    embedDataForm.addEventListener('submit', e=>{
      e.preventDefault();
      const hiddenData = document.getElementById('embed-hidden-data').value;
      const embedMaxCapacity = parseInt(document.getElementById('embed-max-capacity').innerText, 10);
      if(hiddenData.length > embedMaxCapacity){
        alert(`Data too large! Max capacity: ${embedMaxCapacity} chars.`);
        return;
      }
      const embedUploadedFilename = getUploadedFilename();
      if(!embedUploadedFilename){
        alert("No image uploaded for embedding.");
        return;
      }
      fetch('/embed_data',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          filename: embedUploadedFilename,
          hidden_data: hiddenData
        })
      })
      .then(r=>r.json())
      .then(res=>{
        if(res.error){
          alert(res.error);
        }
        else{
          stegoPngFilename = res.stego_png_filename;
          stegoTiffFilename = res.stego_tiff_filename;
          // Update the stego image preview
          stImg.src = `/download/stego/${stegoPngFilename}`;
          stImg.dataset.stegofilePng = stegoPngFilename;
          stImg.dataset.stegofileTiff = stegoTiffFilename;
          stImg.onload = () => {
            // Enable the download button
            downloadButton.disabled = false;
            alert("Data embedded successfully!");
          };
        }
      })
      .catch(err=>console.error("Embed data error:",err));
    });
  }

  // Helper function to get the uploaded filename from the original or camera upload
  function getUploadedFilename(){
    const origImg = document.getElementById('embed-original-image');
    const src = origImg.src;
    if(src.includes('/download/original/')){
      return src.split('/download/original/')[1];
    }
    return '';
  }

  // Function to display extracted data on the front-end
  function displayExtractedData(data){
    const extractedDataElem = document.getElementById('embed-extracted-data');
    const extractedDataSection = document.getElementById('embed-extracted-data-section');
    if(extractedDataElem && extractedDataSection){
      extractedDataElem.innerText = data;
      extractedDataSection.style.display = 'block';
    }
  }

  // Download Stego Image Logic
  if(downloadButton){
    downloadButton.addEventListener('click', ()=>{
      const fmt = downloadFormatSelect.value;
      let stegoFilename = '';
      if(fmt === 'png'){
        stegoFilename = stImg.dataset.stegofilePng;
        if(!stegoFilename){
          alert("Stego PNG image filename not found.");
          return;
        }
        alert("Warning: Downloading as PNG may reduce accuracy due to rounding.");
      }
      else if(fmt === 'tiff'){
        stegoFilename = stImg.dataset.stegofileTiff;
        if(!stegoFilename){
          alert("Stego TIFF image filename not found.");
          return;
        }
      }
      else{
        alert("Invalid format selected.");
        return;
      }

      window.location.href = `/download/stego/${stegoFilename}`;
    });
  }

  // ----------------- EXTRACT SECTION -----------------
  const extractUploadFormElem = document.getElementById('extract-upload-form');
  const extractDetectForm = document.getElementById('extract-detect-form');

  if(extractUploadFormElem){
    extractUploadFormElem.addEventListener('submit', e=>{
      e.preventDefault();
      const formData = new FormData(extractUploadFormElem);
      formData.append('operation','extract');
      fetch('/upload',{
        method:'POST',
        body:formData
      })
      .then(r=>r.json())
      .then(info=>{
        if (info.error) {
          alert(info.error);
        } else {
          const extractStegoFilename = info.filename;
          const extDetImg = document.getElementById("extract-detected-image");
          extDetImg.src = `/download/original/${extractStegoFilename}`;
          fetch(`/download/original/${extractStegoFilename}`)
            .then((response) => {
              if (response.ok) {
                console.log("Extract stego filename:", extractStegoFilename);
                document.getElementById(
                  "extract-detection-section"
                ).style.display = "block";
                alert("Stego image uploaded for extraction!");
              } else {
                console.error("Failed to fetch the image");
              }
            })
            .catch((err) => console.error("Error fetching the image:", err));
        }
      })
      .catch(er=>console.error("Extract upload error:",er));
    });
  }

  if(extractDetectForm){
    extractDetectForm.addEventListener('submit', e=>{
      e.preventDefault();
      const objClass = document.getElementById('extract-object-class').value;
      const extractStegoFilename = getExtractUploadedFilename();
      if(!extractStegoFilename){
        alert("Stego image filename not found.");
        return;
      }
      fetch('/detect',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          filename: extractStegoFilename,
          object_class: objClass,
          operation:'extract'
        })
      })
      .then(r=>r.json())
      .then(res=>{
        if(res.error){
          alert(res.error);
        }
        else{
          const extractDetectedFilename = res.detected_filename;
          const eImg = document.getElementById('extract-detected-image');
          eImg.src = `/download/detected/${extractDetectedFilename}`;
          eImg.onload=()=>{
            document.getElementById('extract-results').style.display='block';
            document.getElementById('extracted-data').innerText='Loading...';
            if(res.detections.length>0){
              let promises = res.detections.map(det=>{
                return fetch('/extract_data',{
                  method:'POST',
                  headers:{'Content-Type':'application/json'},
                  body: JSON.stringify({
                    stego_filename: extractStegoFilename,
                    selected_roi: {
                      x1:det.x1,
                      y1:det.y1,
                      x2:det.x2,
                      y2:det.y2
                    }
                  })
                })
                .then(rr=>rr.json())
                .then(xx=>{
                  if(xx.error){
                    console.warn("Extraction error:", xx.error);
                    return '';
                  }
                  if(xx.extracted_data){
                    document.getElementById('extracted-data').innerText=xx.extracted_data;
                  }
                  else{
                    document.getElementById('extracted-data').innerText='No data extracted.';
                  }
                  alert("Data extracted successfully!");
                  return xx.extracted_data;
                });
              });
            }
            else{
              alert("No objects found for extraction!");
            }
          };
        }
      })
      .catch(er=>console.error("Extract detect error:",er));
    });
  }

  // Helper function to get the uploaded stego filename from the extract upload
  function getExtractUploadedFilename(){
    const extDetImg = document.getElementById('extract-detected-image');
    const src = extDetImg.src;
    if(src.includes('/download/original/')){
      return src.split('/download/original/')[1]; // Includes extension
    }
    return '';
  }

  // ----------------- DOWNLOAD ROUTE FOR EXTRACT PAGE -----------------
  // Assuming download is handled similarly for both embed and extract,
  // no additional changes are needed here.

});
