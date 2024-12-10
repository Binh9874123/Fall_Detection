<View>
  <Header value="Object detection" />
  <Labels name="bbox_labels" toName="video" allowEmpty="false">
    <Label value="A"/>
    <Label value="B"/>
    <Label value="C"/>
  </Labels>

  <VideoRectangle name="bounding_box" toName="video"/>

  <!-- Video source -->
  <Video name="video" value="$video" height="300"/>
</View>