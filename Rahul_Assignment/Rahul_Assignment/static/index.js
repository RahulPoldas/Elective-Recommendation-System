const studentForm = document.getElementById("studentForm");

studentForm.addEventListener("submit", function (e) {
  e.preventDefault(); 
  const rollIndex = parseInt(document.getElementById("studentRollNo").value);
  
  if (isNaN(rollIndex) || rollIndex < 0) {
    alert("Please enter a valid roll number (positive integer).");
    return;
  }

  studentForm.submit();
});
